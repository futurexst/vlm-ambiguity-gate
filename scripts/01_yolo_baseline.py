#!/usr/bin/env python3
"""
01_yolo_baseline.py — VLM 없이 YOLO만 썼을 때의 오탐(FP)·미탐(FN) 기준선 측정

사용법:
  python scripts/01_yolo_baseline.py --seq ~/work/data/MOT17/train/MOT17-02-FRCNN

결과 파일:
  runs/det/<태그>.csv           : 프레임별 YOLO 박스 (신뢰도 0.05 이상 전부)
  runs/eval/<태그>_labeled.csv  : 박스마다 TP/FP/IGN 판정 → 다음 단계 VLM 평가용 데이터
"""
import argparse
import configparser
import csv
import os
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

# MOT17 정답 클래스 중 "사람처럼 보이지만 평가에서 빼는 것"
#   2: 탈것 위의 사람, 7: 멈춰 있는 사람, 8: 방해물(마네킹·포스터 등), 12: 반사
PERSON_LIKE_IGNORE = {2, 7, 8, 12}
DISTRACTOR = {8, 12}   # VLM이 "not person"으로 걸러내야 하는 대표 대상


def load_seqinfo(seq_dir):
    """seqinfo.ini에서 프레임 수, 이미지 폴더 이름, 확장자를 읽는다"""
    cp = configparser.ConfigParser()
    cp.read(os.path.join(seq_dir, "seqinfo.ini"))
    s = cp["Sequence"]
    return int(s["seqLength"]), s["imDir"], s["imExt"]


def load_gt(seq_dir):
    """gt.txt → {프레임: [[x1, y1, x2, y2, id, 사용여부, 클래스, 가시도], ...]}"""
    gt = {}
    with open(os.path.join(seq_dir, "gt", "gt.txt")) as f:
        for r in csv.reader(f):
            fr = int(r[0])
            x, y, w, h = map(float, r[2:6])
            gt.setdefault(fr, []).append([x, y, x + w, y + h, int(r[1]), int(r[6]), int(r[7]), float(r[8])])
    return gt


def iou(a, b):
    """박스 목록 a(N개)와 b(M개)의 겹침 비율(IoU) 행렬 N×M"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = np.asarray(a, float)[:, :4]
    b = np.asarray(b, float)[:, :4]
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(dets, gts, thr):
    """헝가리안 매칭(전체 겹침이 가장 커지게 1:1 짝짓기). IoU ≥ thr인 쌍만 반환"""
    m = iou(dets, gts)
    if m.size == 0:
        return []
    r, c = linear_sum_assignment(-m)
    return [(i, j) for i, j in zip(r, c) if m[i, j] >= thr]


def label_frame(dets, gts, iou_thr):
    """한 프레임의 YOLO 박스마다 TP/FP/IGN을 정하고, 놓친 정답 수(FN)를 센다"""
    pos = [g for g in gts if g[5] == 1 and g[6] == 1]                       # 평가 대상 보행자
    ign = [g for g in gts if not (g[5] == 1 and g[6] == 1) and g[6] in PERSON_LIKE_IGNORE]
    status = ["FP"] * len(dets)
    gcls = [0] * len(dets)
    pairs = match(dets, pos, iou_thr)
    for i, _ in pairs:
        status[i] = "TP"
        gcls[i] = 1
    rest = [i for i in range(len(dets)) if status[i] == "FP"]                # 보행자와 안 겹친 박스
    for k, j in match([dets[i] for i in rest], ign, iou_thr):               # 평가 제외 대상과 겹치면 IGN
        i = rest[k]
        status[i] = "IGN"
        gcls[i] = ign[j][6]
    return status, gcls, len(pos) - len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="MOT17 영상 폴더 (예: .../MOT17-02-FRCNN)")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=1280, help="YOLO 입력 크기. 1080p 영상의 작은 사람을 잡으려고 1280")
    ap.add_argument("--iou", type=float, default=0.5, help="정답과 같은 박스로 볼 최소 겹침")
    ap.add_argument("--rerun", action="store_true", help="저장된 탐지 결과가 있어도 YOLO를 다시 돌림")
    a = ap.parse_args()

    seq_dir = os.path.expanduser(a.seq).rstrip("/")
    seq = os.path.basename(seq_dir)
    tag = f"{seq}_{os.path.splitext(os.path.basename(a.model))[0]}_{a.imgsz}"
    os.makedirs("runs/det", exist_ok=True)
    os.makedirs("runs/eval", exist_ok=True)
    det_csv = f"runs/det/{tag}.csv"
    n, imdir, ext = load_seqinfo(seq_dir)

    # ── 1) YOLO 탐지 (결과가 이미 있으면 건너뜀 → 평가 기준만 바꿔 다시 볼 때 빠름) ──
    if a.rerun or not os.path.exists(det_csv):
        from ultralytics import YOLO
        model = YOLO(a.model)
        times = []
        with open(det_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "x1", "y1", "x2", "y2", "conf"])
            for fr in range(1, n + 1):
                img = os.path.join(seq_dir, imdir, f"{fr:06d}{ext}")
                t = time.perf_counter()
                r = model.predict(img, imgsz=a.imgsz, conf=0.05, classes=[0], verbose=False)[0]
                times.append((time.perf_counter() - t) * 1000)
                for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                    w.writerow([fr, round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1), round(c, 4)])
        ts = np.array(times[5:])   # 앞 5프레임은 워밍업이라 제외
        print(f"YOLO {a.model} imgsz={a.imgsz}: 프레임당 중앙값 {np.median(ts):.1f}ms, p90 {np.percentile(ts, 90):.1f}ms (이미지 읽기 포함)")

    # ── 2) 정답과 비교 ──
    dets = {}
    with open(det_csv) as f:
        for r in csv.DictReader(f):
            dets.setdefault(int(r["frame"]), []).append(
                [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]), float(r["conf"])])
    gt = load_gt(seq_dir)

    print(f"\n[{seq}] 프레임 {n}개, IoU ≥ {a.iou}")
    print(f"{'conf≥':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'IGN':>5} {'정밀도':>7} {'재현율':>7}")
    for thr in [0.1, 0.25, 0.4, 0.5, 0.6]:
        TP = FP = FN = IG = 0
        for fr in range(1, n + 1):
            d = [x for x in dets.get(fr, []) if x[4] >= thr]
            st, _, fn = label_frame(d, gt.get(fr, []), a.iou)
            TP += st.count("TP")
            FP += st.count("FP")
            IG += st.count("IGN")
            FN += fn
        print(f"{thr:>6.2f} {TP:>6} {FP:>6} {FN:>6} {IG:>5} {TP / max(TP + FP, 1):>7.3f} {TP / max(TP + FN, 1):>7.3f}")

    # ── 3) 박스별 판정 저장 + 신뢰도 구간별 오탐 비율 ──
    rows = []
    for fr in range(1, n + 1):
        d = dets.get(fr, [])
        st, gc, _ = label_frame(d, gt.get(fr, []), a.iou)
        for x, s, g in zip(d, st, gc):
            rows.append([fr] + x + [s, g])          # frame, x1, y1, x2, y2, conf, status, gt_cls
    lab_csv = f"runs/eval/{tag}_labeled.csv"
    with open(lab_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "x1", "y1", "x2", "y2", "conf", "status", "gt_cls"])
        w.writerows(rows)

    print("\n신뢰도 구간별 박스 상태 — VLM을 부를 '애매한 구간' 찾기")
    print(f"{'conf 구간':>11} {'TP':>6} {'FP':>6} {'IGN':>5} {'오탐비율':>8}")
    for lo, hi in [(0.05, 0.25), (0.25, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        sel = [r for r in rows if lo <= r[5] < hi]
        tp = sum(r[6] == "TP" for r in sel)
        fp = sum(r[6] == "FP" for r in sel)
        ig = sum(r[6] == "IGN" for r in sel)
        print(f"{lo:.2f}~{min(hi, 1.0):.2f} {tp:>6} {fp:>6} {ig:>5} {fp / max(len(sel), 1):>8.1%}")
    dist = sum(1 for r in rows if r[6] == "IGN" and r[7] in DISTRACTOR)
    print(f"\n방해물·반사(class 8·12)에 걸린 박스: {dist}개")
    print(f"결과 파일: {lab_csv}")


if __name__ == "__main__":
    main()
