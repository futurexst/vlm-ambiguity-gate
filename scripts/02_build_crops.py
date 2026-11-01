#!/usr/bin/env python3
"""
02_build_crops.py — 공통 시험지(test)·학습 데이터(train)용 크롭 만들기

대상 : YOLO 신뢰도 0.25 이상 0.6 미만의 '애매 구간' 박스
정답 :
  person     : TP — 평가 대상 보행자와 IoU ≥ 0.5
  not_person : ① FP 중 어떤 정답 박스와도 IoU < 0.1 (배경·물체 오탐)
               ② 평가 제외 박스 중 방해물·반사 (MOT17 class 8·12)
제외 : 박스 어긋남 FP(정답과 IoU 0.1~0.5), 멈춘 사람·탈것 위 사람(class 2·7) → 정답이 애매해서 뺌

사용법:
  python scripts/02_build_crops.py --split test  --seqs MOT17-02-FRCNN --per-class 200
  python scripts/02_build_crops.py --split train --seqs MOT17-04-FRCNN MOT17-09-FRCNN --per-class 600
결과:
  ~/work/data/crops/<split>/<label>/*.jpg
  ~/work/data/crops/<split>.csv   (크롭 목록 + 정답)
"""
import argparse
import csv
import importlib
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
B = importlib.import_module("01_yolo_baseline")   # load_gt, load_seqinfo, iou, DISTRACTOR 재사용


def collect(seq_dir, lab_csv, lo, hi):
    """판정 CSV에서 애매 구간 박스를 골라 정답(person / not_person)을 붙인다"""
    gt = B.load_gt(seq_dir)
    pool = {"person": [], "not_person": []}
    with open(lab_csv) as f:
        for r in csv.DictReader(f):
            c = float(r["conf"])
            if not (lo <= c < hi):
                continue
            fr = int(r["frame"])
            box = [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])]
            st, gcls = r["status"], int(r["gt_cls"])
            if st == "TP":
                label, src = "person", "TP"
            elif st == "IGN" and gcls in B.DISTRACTOR:
                label, src = "not_person", f"distractor_c{gcls}"
            elif st == "FP":
                g = gt.get(fr, [])
                if g and B.iou([box], g).max() >= 0.1:
                    continue                      # 박스 어긋남 → 제외
                label, src = "not_person", "background"
            else:
                continue                          # 멈춘 사람 등 → 제외
            pool[label].append({"frame": fr, "box": box, "conf": c, "src": src})
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--per-class", type=int, default=200, help="정답별 최대 개수 (여러 영상 합계)")
    ap.add_argument("--lo", type=float, default=0.25)
    ap.add_argument("--hi", type=float, default=0.6)
    ap.add_argument("--pad", type=float, default=0.1, help="박스 주변 여백 비율")
    ap.add_argument("--data", default="~/work/data/MOT17/train")
    ap.add_argument("--out", default="~/work/data/crops")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    data, out = os.path.expanduser(a.data), os.path.expanduser(a.out)
    items = {"person": [], "not_person": []}
    for seq in a.seqs:
        pool = collect(os.path.join(data, seq), f"runs/eval/{seq}_yolo11n_1280_labeled.csv", a.lo, a.hi)
        for k in items:
            for it in pool[k]:
                it["seq"] = seq
            items[k] += pool[k]
        print(f"{seq}: 후보 person {len(pool['person'])}, not_person {len(pool['not_person'])}")

    rng = random.Random(a.seed)                    # 시드 고정 → 다시 돌려도 같은 시험지
    chosen = []
    for k in items:
        sel = items[k] if len(items[k]) <= a.per_class else rng.sample(items[k], a.per_class)
        for it in sel:
            it["label"] = k
        chosen += sel

    by_frame = defaultdict(list)                   # 같은 프레임은 이미지를 한 번만 연다
    for it in chosen:
        by_frame[(it["seq"], it["frame"])].append(it)

    rows = []
    for (seq, fr), its in sorted(by_frame.items()):
        _, imdir, ext = B.load_seqinfo(os.path.join(data, seq))
        im = Image.open(os.path.join(data, seq, imdir, f"{fr:06d}{ext}")).convert("RGB")
        W, H = im.size
        for i, it in enumerate(its):
            x1, y1, x2, y2 = it["box"]
            px, py = (x2 - x1) * a.pad, (y2 - y1) * a.pad
            cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
            cx2, cy2 = min(W, int(x2 + px)), min(H, int(y2 + py))
            if cx2 - cx1 < 4 or cy2 - cy1 < 4:
                continue
            d = os.path.join(out, a.split, it["label"])
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{seq}_{fr:06d}_{i}.jpg")
            im.crop((cx1, cy1, cx2, cy2)).save(path, quality=95)
            rows.append([path, it["label"], seq, fr, round(it["conf"], 4), it["src"],
                         round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1), round(y2 - y1, 1)])

    man = os.path.join(out, f"{a.split}.csv")
    with open(man, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "seq", "frame", "conf", "src", "x1", "y1", "x2", "y2", "box_h"])
        w.writerows(rows)

    print(f"\n[{a.split}] 크롭 {len(rows)}장 → {man}")
    print("  정답별:", dict(Counter(r[1] for r in rows)))
    print("  출처별:", dict(Counter(r[5] for r in rows)))
    hs = np.array([r[10] for r in rows])
    print(f"  박스 높이(px): 중앙값 {np.median(hs):.0f}, 최소 {hs.min():.0f}, 최대 {hs.max():.0f}")


if __name__ == "__main__":
    main()
