#!/usr/bin/env python3
"""
06_sweep_prompt_input.py — 질문 문구 × 크롭 입력 방식을 검증 영상(MOT17-10)으로 제로샷 비교
※ 시험지(MOT17-02)는 쓰지 않는다. 고르는 과정에 시험지가 섞이면 점수가 부풀려지므로 최종 1회 채점용으로 남겨 둔다.

사용법:
  python scripts/06_sweep_prompt_input.py --manifest ~/work/data/crops/val10.csv
  python scripts/06_sweep_prompt_input.py --manifest ~/work/data/crops/val10.csv --prompts P0 P1 --inputs I0 I2
결과:
  runs/sweep/<입력>_<문구>.csv (크롭별 점수), runs/sweep/summary.csv (조합별 지표)
"""
import argparse
import csv
import importlib
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.expanduser("~/work/OmniJev"))
from mso.infer import MSO1  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
E = importlib.import_module("03_eval_omnijev")   # auroc
C = importlib.import_module("05_compare")        # removal_at
from prompts import PROMPTS                       # noqa: E402

# 입력 방식: pad = 박스 주변 여백 비율, min_h = 크롭 높이가 이보다 작으면 이 높이로 확대 (0이면 확대 안 함)
INPUTS = {
    "I0": {"pad": 0.1, "min_h": 0},
    "I1": {"pad": 0.1, "min_h": 336},
    "I2": {"pad": 0.3, "min_h": 336},
}


def make_image(r, cfg, data, out_dir):
    """원본 프레임에서 입력 방식대로 다시 잘라 저장 (이미 있으면 재사용)"""
    path = os.path.join(out_dir, os.path.basename(r["path"]))
    if os.path.exists(path):
        return path
    x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
    im = Image.open(os.path.join(data, r["seq"], "img1", f"{int(r['frame']):06d}.jpg")).convert("RGB")
    W, H = im.size
    px, py = (x2 - x1) * cfg["pad"], (y2 - y1) * cfg["pad"]
    c = im.crop((max(0, int(x1 - px)), max(0, int(y1 - py)), min(W, int(x2 + px)), min(H, int(y2 + py))))
    if cfg["min_h"] and c.height < cfg["min_h"]:
        s = cfg["min_h"] / c.height
        c = c.resize((max(1, round(c.width * s)), cfg["min_h"]), Image.LANCZOS)
    c.save(path, quality=95)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--prompts", nargs="+", default=list(PROMPTS))
    ap.add_argument("--inputs", nargs="+", default=list(INPUTS))
    ap.add_argument("--data", default="~/work/data/MOT17/train")
    ap.add_argument("--ckpt", default="~/work/models/omnijev-0.8b")
    ap.add_argument("--base", default="~/work/models/qwen3.5-0.8b")
    a = ap.parse_args()

    data = os.path.expanduser(a.data)
    rows = list(csv.DictReader(open(os.path.expanduser(a.manifest))))
    y = np.array([r["label"] == "not_person" for r in rows], int)
    m = MSO1(os.path.expanduser(a.ckpt), os.path.expanduser(a.base))
    os.makedirs("runs/sweep", exist_ok=True)
    tag = os.path.splitext(os.path.basename(a.manifest))[0]
    print(f"[{tag}] {len(rows)}장 (사람 {int((y == 0).sum())} / 사람 아님 {int(y.sum())}), "
          f"조합 {len(a.prompts) * len(a.inputs)}개")

    summary = []
    for ik in a.inputs:
        out_dir = os.path.expanduser(f"~/work/data/crops/sweep_{tag}_{ik}")
        os.makedirs(out_dir, exist_ok=True)
        paths = [make_image(r, INPUTS[ik], data, out_dir) for r in rows]
        for pk in a.prompts:
            Q = PROMPTS[pk]
            m.system_one({"images": [paths[0]]}, Q)                 # 워밍업
            t0 = time.time()
            p_not, noul, pred = [], [], []
            for p in paths:
                ans = m.system_one({"images": [p]}, Q)
                p_not.append(ans["what"]["probabilities"]["not person"])
                noul.append(ans["real"]["noul"])
                pred.append(int(ans["what"]["choice"] == "not person"))
            p_not, noul, pred = np.array(p_not), np.array(noul), np.array(pred)
            combo = (p_not + (1 - noul)) / 2                         # 두 질문 결합 점수
            res = {"input": ik, "prompt": pk,
                   "acc": (pred == y).mean(), "person_acc": (pred[y == 0] == 0).mean(),
                   "auroc_choice": E.auroc(p_not, y), "auroc_noul": E.auroc(1 - noul, y),
                   "auroc_combo": E.auroc(combo, y),
                   "rem5_choice": C.removal_at(p_not, y, 0.05)[0], "rem5_combo": C.removal_at(combo, y, 0.05)[0],
                   "sec": round(time.time() - t0)}
            summary.append(res)
            print(f"  {ik}·{pk}: 정확도 {res['acc']:.3f} 사람정답 {res['person_acc']:.3f} | AUROC 선택 "
                  f"{res['auroc_choice']:.3f} 참거짓 {res['auroc_noul']:.3f} 결합 {res['auroc_combo']:.3f} | "
                  f"제거@손실5% 선택 {res['rem5_choice']:.1%} 결합 {res['rem5_combo']:.1%} ({res['sec']}초)")
            with open(f"runs/sweep/{tag}_{ik}_{pk}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["path", "label", "p_not", "noul_real", "pred_not"])
                w.writerows(zip(paths, [r["label"] for r in rows], p_not, noul, pred))

    with open(f"runs/sweep/{tag}_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print("\n결합 AUROC 높은 순")
    for s in sorted(summary, key=lambda s: -s["auroc_combo"]):
        print(f"  {s['input']}·{s['prompt']}: 결합 AUROC {s['auroc_combo']:.3f}, 제거@손실5% {s['rem5_combo']:.1%}, "
              f"사람정답 {s['person_acc']:.3f}")


if __name__ == "__main__":
    main()
