#!/usr/bin/env python3
"""
03_eval_omnijev.py — 공통 시험지 크롭으로 OmniJev 판정 정확도 채점

사용법:
  python scripts/03_eval_omnijev.py --manifest ~/work/data/crops/test.csv --tag zeroshot
  python scripts/03_eval_omnijev.py --manifest ~/work/data/crops/test.csv --tag headft --head runs/ft/head_ft.pt
결과:
  runs/eval/omnijev_<tag>.csv (크롭별 확률) + 화면 지표 표
채점 기준:
  양성 = not_person (YOLO 오탐 → 지워야 할 박스)
  오탐 제거율      = not_person 중 지운 비율   (높을수록 좋음)
  사람 잘못 지움   = person 중 지운 비율      (낮을수록 좋음, 미탐을 새로 만드는 비율)
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import rankdata

sys.path.insert(0, os.path.expanduser("~/work/OmniJev"))
from mso.infer import MSO1  # noqa: E402

QUESTIONS = {
    "what": {"type": "choice",
             "instructions": "What is shown in this image?",
             "criteria": {
                 "person":          "a fully visible real person",
                 "occluded person": "a real person that is partly hidden",
                 "not person":      "not a real person: mannequin, poster, shadow, reflection or another object"}},
    "real": {"type": "noul", "instructions": "There is a real person in this image."},
}


def auroc(score, y):
    """순위 기반 AUROC: 무작위로 고른 양성이 음성보다 점수가 높을 확률"""
    score, y = np.asarray(score, float), np.asarray(y, int)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(p, y, bins=10):
    """보정 오차: 확률 구간별 '말한 확률'과 '실제 맞은 비율'의 차이 평균"""
    p, y = np.asarray(p, float), np.asarray(y, int)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--head", default=None, help="미세조정한 판정 헤드 가중치(.pt). 없으면 원본 그대로")
    ap.add_argument("--ckpt", default="~/work/models/omnijev-0.8b")
    ap.add_argument("--base", default="~/work/models/qwen3.5-0.8b")
    a = ap.parse_args()

    m = MSO1(os.path.expanduser(a.ckpt), os.path.expanduser(a.base))
    if a.head:
        m.model.head.load_state_dict(torch.load(a.head, map_location=m.dev))
        print(f"판정 헤드 교체: {a.head}")
    torch.cuda.reset_peak_memory_stats()

    rows = list(csv.DictReader(open(os.path.expanduser(a.manifest))))
    m.system_one({"images": [rows[0]["path"]]}, QUESTIONS)          # 워밍업
    torch.cuda.synchronize()

    out = []
    for i, r in enumerate(rows):
        t = time.perf_counter()
        ans = m.system_one({"images": [r["path"]]}, QUESTIONS)      # 매번 새 크롭 → 전처리 포함 실제 지연
        torch.cuda.synchronize()
        lat = (time.perf_counter() - t) * 1000
        pr = ans["what"]["probabilities"]
        out.append({"path": r["path"], "label": r["label"], "src": r["src"], "conf": r["conf"],
                    "p_person": pr["person"], "p_occluded": pr["occluded person"], "p_not": pr["not person"],
                    "abstain": ans["what"]["abstain"], "confidence": ans["what"]["confidence"],
                    "choice": ans["what"]["choice"], "noul_real": ans["real"]["noul"], "lat_ms": round(lat, 1)})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)}")

    os.makedirs("runs/eval", exist_ok=True)
    res_csv = f"runs/eval/omnijev_{a.tag}.csv"
    with open(res_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    y = np.array([o["label"] == "not_person" for o in out], int)
    p_not = np.array([o["p_not"] for o in out])
    pred = np.array([o["choice"] == "not person" for o in out], int)
    lat = np.array([o["lat_ms"] for o in out])

    print(f"\n[{a.tag}] 시험지 {len(out)}장 (person {int((y == 0).sum())}, not_person {int(y.sum())})")
    print(f"  정확도(최고 확률 선택지 기준) : {(pred == y).mean():.3f}")
    print(f"  AUROC (p_not)                  : {auroc(p_not, y):.3f}")
    print(f"  AUROC (1 - noul 'real person') : {auroc(1 - np.array([o['noul_real'] for o in out]), y):.3f}")
    print(f"  ECE (p_not)                    : {ece(p_not, y):.3f}")
    print(f"  지연(질문 2개, 전처리 포함)    : 중앙값 {np.median(lat):.0f}ms, p90 {np.percentile(lat, 90):.0f}ms")
    print(f"  최대 VRAM                      : {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GB")

    print("\n  p_not 기준  오탐 제거율  사람 잘못 지움")
    for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
        print(f"    ≥ {t:.1f}      {(p_not[y == 1] >= t).mean():>8.1%}      {(p_not[y == 0] >= t).mean():>8.1%}")

    print("\n  출처별 정확도")
    g = defaultdict(list)
    for o, yy, pp in zip(out, y, pred):
        g[o["src"]].append(yy == pp)
    for k, v in sorted(g.items()):
        print(f"    {k:<16} {np.mean(v):.3f}  (n={len(v)})")
    print(f"\n결과 파일: {res_csv}")


if __name__ == "__main__":
    main()
