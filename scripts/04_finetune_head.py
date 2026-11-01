#!/usr/bin/env python3
"""
04_finetune_head.py — OmniJev 판정 헤드 미세조정 (백본 Qwen3.5 + LoRA는 고정)

흐름:
  1) 특징 추출: 크롭마다 백본+LoRA를 한 번 통과시켜 판정 헤드의 입력을 저장
       u     : 선택지마다 백본이 만든 표현 [K, D]
       zq    : 질문 표현 [D]
       feats : 백본이 선택지를 얼마나 그럴듯하게 보는지 [K, 6]
  2) 학습: 저장한 입력으로 판정 헤드만 학습 (적정 채점 규칙: 로그 점수)
  3) 저장: runs/ft/head_ft.pt → 03_eval_omnijev.py --head 로 시험지 채점

사용법:
  python scripts/04_finetune_head.py --train ~/work/data/crops/train.csv
"""
import argparse
import copy
import csv
import importlib
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/work/OmniJev"))
from mso.infer import MSO1           # noqa: E402
from mso import branch as BR         # noqa: E402
from mso.head import noul_loss       # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
E = importlib.import_module("03_eval_omnijev")   # QUESTIONS, auroc 재사용 (채점과 같은 질문)
QUESTIONS = E.QUESTIONS


@torch.no_grad()
def extract(m, path):
    """infer.py의 ask_branch와 같은 순서로 백본을 통과시켜, 헤드 직전의 입력만 꺼낸다"""
    qids = list(QUESTIONS)
    keys_l, opts_l, ids_l, spans_l, pix = m._encode_many(path, None, QUESTIONS, qids)
    L, mn = 0, min(len(x) for x in ids_l)
    while L < mn and all(x[L] == ids_l[0][L] for x in ids_l):
        L += 1
    L = max(1, min([L] + [sp[0][0] - 1 for sp in spans_l if sp[0]]))
    base = dict(pix)
    base["input_ids"] = torch.tensor(ids_l[0], device=m.dev)[None]
    base["attention_mask"] = torch.ones_like(base["input_ids"])
    penc = BR.prefix_inputs(base, L)
    ppos = BR.prefix_positions(m.model, penc)
    qrows = [BR.rows_from_ids(row, L, sp[0], sp[1], k=len(opts))
             for row, sp, opts in zip(ids_l, spans_l, opts_l)]
    outs = BR.branch_questions(m.model.backbone, penc, qrows, m.dev, m.open_ids, m.close_ids,
                               lm_head=m.lm, prefix_pos=ppos, pad_id=m.pad_id)
    res = {}
    for qid, opts, (u, zq, feats) in zip(qids, opts_l, outs):
        k = min(int(u.shape[0]), len(opts))
        res[qid] = (u[:k].float().cpu(), zq.float().cpu(),
                    feats[:k].float().cpu() if feats is not None else None)
    return res


def head_mu(head, feat, type_id, dev):
    u, zq, f = feat
    return head(u.to(dev), zq.to(dev), type_id, f.to(dev) if f is not None else None)


def sample_loss(head, s, dev, i_not):
    """사람: -log(사람+가려진 사람) / 사람 아님: -log(사람 아님) + 참/거짓 로그 손실 + 확률합 초과 벌점"""
    mu = head_mu(head, s["what"], 1, dev)                       # [3] person, occluded, not person
    p_not = mu[i_not]
    p_per = mu.sum() - p_not
    lc = -torch.log(p_not.clamp(min=1e-6)) if s["y"] == 1 else -torch.log(p_per.clamp(min=1e-6))
    excess = torch.relu(mu.sum() - 1.0)
    mu_real = head_mu(head, s["real"], 0, dev)[0]               # P("There is a real person")
    ln = noul_loss(mu_real, torch.tensor(1.0 - s["y"], device=dev))
    return lc + ln + excess


@torch.no_grad()
def evaluate(head, data, dev, i_not):
    head.eval()
    losses, y, p_not, pred = [], [], [], []
    for s in data:
        losses.append(float(sample_loss(head, s, dev, i_not)))
        mu = head_mu(head, s["what"], 1, dev)
        y.append(s["y"])
        p_not.append(float(mu[i_not]))
        pred.append(int(int(torch.argmax(mu)) == i_not))
    y, pred = np.array(y), np.array(pred)
    return {"loss": float(np.mean(losses)), "acc": float((pred == y).mean()),
            "auroc": E.auroc(np.array(p_not), y),
            "person_acc": float((pred[y == 0] == 0).mean()), "not_acc": float((pred[y == 1] == 1).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--ckpt", default="~/work/models/omnijev-0.8b")
    ap.add_argument("--base", default="~/work/models/qwen3.5-0.8b")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reextract", action="store_true")
    ap.add_argument("--val-seq", default=None, help="이 영상을 통째로 검증용으로 사용 (정보 누수 방지)")
    ap.add_argument("--out", default="runs/ft/head_ft.pt")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs("runs/ft", exist_ok=True)
    m = MSO1(os.path.expanduser(a.ckpt), os.path.expanduser(a.base))
    assert m.branch, "Qwen3.5 하이브리드 백본(branch 경로)을 가정한 스크립트입니다"
    dev, head = m.dev, m.model.head
    keys, _ = m._options(QUESTIONS["what"])
    i_not = keys.index("not person")
    rows = list(csv.DictReader(open(os.path.expanduser(a.train))))

    # ── 0) 확인: 저장한 입력 + 원본 헤드 = 실제 추론 결과인가 ──
    feat0 = extract(m, rows[0]["path"])
    ans = m.system_one({"images": [rows[0]["path"]]}, QUESTIONS)
    keys_w, opts_w = m._options(QUESTIONS["what"])
    with torch.no_grad():
        fin = m._finish(QUESTIONS["what"], keys_w, opts_w, head_mu(head, feat0["what"], 1, dev), 0.0)
    a1, a2 = ans["what"]["probabilities"]["not person"], fin["probabilities"]["not person"]
    print(f"[확인] 실제 추론 p_not={a1:.4f} / 저장 입력으로 계산 p_not={a2:.4f}")
    if abs(a1 - a2) > 0.02:
        sys.exit("두 값이 다릅니다. 이 출력을 그대로 보내주세요.")

    # ── 1) 특징 추출 (한 번만, 이후엔 저장본 사용) ──
    cache = f"runs/ft/feats_{os.path.splitext(os.path.basename(a.train))[0]}.pt"
    if os.path.exists(cache) and not a.reextract:
        data = torch.load(cache)
        print(f"저장된 특징 사용: {cache} ({len(data)}개)")
    else:
        data, t0 = [], time.time()
        for i, r in enumerate(rows):
            f = extract(m, r["path"])
            data.append({"what": f["what"], "real": f["real"], "y": int(r["label"] == "not_person"),
                         "src": r["src"], "seq": r["seq"]})
            if (i + 1) % 100 == 0:
                print(f"  특징 추출 {i + 1}/{len(rows)} ({time.time() - t0:.0f}s)")
        torch.save(data, cache)
        print(f"특징 저장: {cache}")

    # ── 2) 학습/검증 분할 (정답 비율 유지) ──
    idx = {0: [i for i, s in enumerate(data) if s["y"] == 0], 1: [i for i, s in enumerate(data) if s["y"] == 1]}
    val_idx = []
    if a.val_seq:
        val_idx = [i for i, s in enumerate(data) if s["seq"] == a.val_seq]
    else:
        for k in idx:
            random.shuffle(idx[k])
            val_idx += idx[k][:int(len(idx[k]) * a.val_frac)]
    vs = set(val_idx)
    tr = [data[i] for i in range(len(data)) if i not in vs]
    va = [data[i] for i in val_idx]
    n_params = sum(p.numel() for p in head.parameters())
    print(f"\n학습 {len(tr)} / 검증 {len(va)}, 학습 파라미터 {n_params / 1e6:.2f}M (판정 헤드만)")

    for p in head.parameters():
        p.requires_grad_(True)
    zs = evaluate(head, va, dev, i_not)
    print(f"[제로샷 헤드] 검증 loss {zs['loss']:.3f} 정확도 {zs['acc']:.3f} AUROC {zs['auroc']:.3f} "
          f"(사람 {zs['person_acc']:.3f} / 사람아님 {zs['not_acc']:.3f})")

    # ── 3) 학습 ──
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.wd)
    best, best_state, bad, log = float("inf"), None, 0, []
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        head.train()
        random.shuffle(tr)
        tl = []
        for b in range(0, len(tr), a.batch):
            batch = tr[b:b + a.batch]
            loss = torch.stack([sample_loss(head, s, dev, i_not) for s in batch]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl.append(float(loss.detach()))
        v = evaluate(head, va, dev, i_not)
        log.append({"epoch": ep, "train_loss": float(np.mean(tl)), **v})
        print(f"에폭 {ep:>2}: 학습 loss {np.mean(tl):.3f} | 검증 loss {v['loss']:.3f} 정확도 {v['acc']:.3f} "
              f"AUROC {v['auroc']:.3f} (사람 {v['person_acc']:.3f} / 사람아님 {v['not_acc']:.3f})")
        if v["loss"] < best - 1e-4:
            best, best_state, bad = v["loss"], copy.deepcopy(head.state_dict()), 0
        else:
            bad += 1
            if bad >= a.patience:
                print(f"조기 종료 (검증 loss {a.patience}에폭 연속 개선 없음)")
                break

    torch.save(best_state, a.out)
    cfg = {k: v for k, v in vars(a).items()}
    cfg.update({"n_train": len(tr), "n_val": len(va), "trainable_params": n_params,
                "best_val_loss": best, "train_sec": round(time.time() - t0, 1), "zeroshot_val": zs, "log": log})
    json.dump(cfg, open(a.out.replace(".pt", "_log.json"), "w"), ensure_ascii=False, indent=2)
    print(f"\n저장: {a.out} (최적 검증 loss {best:.3f}), 학습 시간 {time.time() - t0:.0f}초")
    print("학습 기록: runs/ft/train_log.json")


if __name__ == "__main__":
    main()
