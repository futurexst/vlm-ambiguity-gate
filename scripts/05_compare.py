#!/usr/bin/env python3
"""
05_compare.py — 여러 채점 결과(runs/eval/omnijev_<tag>.csv)를 같은 기준으로 비교

핵심 기준: "진짜 사람을 최대 f%까지만 잘못 지울 때, 오탐을 몇 % 지우나"
  → 모델마다 확률 눈금이 달라도 공정하게 비교 가능 (기준값을 모델별로 따로 정함)
사용법:
  python scripts/05_compare.py zeroshot headft headft6 headft6_auroc
"""
import csv
import importlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
E = importlib.import_module("03_eval_omnijev")


def removal_at(p_not, y, max_person_loss):
    pp, pn = p_not[y == 0], p_not[y == 1]
    t = np.quantile(pp, 1 - max_person_loss)          # 사람 중 상위 f%만 기준을 넘도록
    return float((pn > t).mean()), float((pp > t).mean()), float(t)


def main():
    tags = sys.argv[1:]
    print(f"{'모델':<16}{'정확도':>7}{'사람정답':>9}{'AUROC':>7}{'ECE':>7}"
          f"{'제거@손실5%':>12}{'제거@손실10%':>13}{'지연중앙값':>10}")
    for tag in tags:
        rows = list(csv.DictReader(open(f"runs/eval/omnijev_{tag}.csv")))
        y = np.array([r["label"] == "not_person" for r in rows], int)
        p = np.array([float(r["p_not"]) for r in rows])
        pred = np.array([r["choice"] == "not person" for r in rows], int)
        lat = np.median([float(r["lat_ms"]) for r in rows])
        r5, _, _ = removal_at(p, y, 0.05)
        r10, _, _ = removal_at(p, y, 0.10)
        print(f"{tag:<16}{(pred == y).mean():>7.3f}{(pred[y == 0] == 0).mean():>9.3f}{E.auroc(p, y):>7.3f}"
              f"{E.ece(p, y):>7.3f}{r5:>12.1%}{r10:>13.1%}{lat:>8.0f}ms")


if __name__ == "__main__":
    main()
