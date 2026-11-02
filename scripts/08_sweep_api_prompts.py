#!/usr/bin/env python3
"""
08_sweep_api_prompts.py — OpenAI Decisions API용 질문 문구 탐색 (검증 영상으로만 선택 → 시험 영상 1회)

왜: 07번에서 쓴 P1은 OmniJev 기준으로 고른 문구라 API에는 "사람" 쪽으로 치우쳤다.
    API에 맞는 문구를 검증 영상(MOT17-10)에서 고르고, 고른 하나만 시험 영상(MOT17-02)에 1회 적용한다.

후보 문구 세트 (각 세트는 질문 1~2개, 점수 계산법 1~3개)
  p1        : 07번과 같은 P1 (기준)
  p0        : 처음 문구 P0
  p3        : 가장 짧은 문구 P3
  verify2   : "탐지가 맞나 / 오탐인가" 선택지 2개
  fa_pred   : "오탐이다" 참/거짓 + "진짜 사람이다" 참/거짓 (두 질문 평균도 계산)
  score5    : 0(확실히 사람) ~ 4(확실히 사람 아님) 5단계 점수 → 확률이 0.01 단위로 끊기는 문제 완화 기대

사용법
  python scripts/08_sweep_api_prompts.py --split val                 # 1단계: 모든 후보를 검증 417장에 (이어 하기 가능)
  python scripts/08_sweep_api_prompts.py --split val --sets p0 score5 # 일부만
  python scripts/08_sweep_api_prompts.py --final                     # 2단계: 검증 AUROC 1등 (세트, 점수법)만 시험 400장에 1회
결과
  runs/api_sweep/val_<세트>.csv, runs/api_sweep/val_summary.csv
  최종: runs/eval/omnijev_val10_openai_best.csv, runs/eval/omnijev_openai_best.csv (05_compare.py 형식)
"""
import argparse
import csv
import importlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import PROMPTS  # noqa: E402

A = importlib.import_module("07_eval_api")   # post_json, b64_file, crop_list, load_env, report, ROOT
E = A.E
C = A.C
ROOT = A.ROOT
URL = A.PROVIDERS["openai"]["url"]
CONTEXT = ("This image is a crop around a box from a pedestrian detector running on street video. "
           "The detector was unsure about this box.")

NOT_DESC = ("no real person in the box: background, an object, a mannequin, a poster, a shadow, "
            "a reflection, or only a tiny fragment of a person at the edge")
PERSON_DESC = "a real person (pedestrian) is the main content of the box, even if small, blurry, partly cut off or partly occluded"


def from_prompt(pname):
    w, r = PROMPTS[pname]["what"], PROMPTS[pname]["real"]
    return [{"type": "choice", "name": "what", "instructions": w["instructions"],
             "choices": [{"value": k, "description": v} for k, v in w["criteria"].items()]},
            {"type": "predicate", "name": "real", "instructions": r["instructions"]}]


# 세트마다: questions(OpenAI 형식), scorers(이름 → 답 사전에서 "사람 아님" 점수 계산), choice_q/not_values(정확도용)
SETS = {
    "p1": {"questions": from_prompt("P1"), "choice_q": "what", "not_values": ["not person"],
           "scorers": {"choice": ("choice", "what", ["not person"]), "pred": ("pred_neg", "real")}},
    "p0": {"questions": from_prompt("P0"), "choice_q": "what", "not_values": ["not person"],
           "scorers": {"choice": ("choice", "what", ["not person"]), "pred": ("pred_neg", "real")}},
    "p3": {"questions": from_prompt("P3"), "choice_q": "what", "not_values": ["not person"],
           "scorers": {"choice": ("choice", "what", ["not person"]), "pred": ("pred_neg", "real")}},
    "verify2": {"questions": [
        {"type": "choice", "name": "verdict",
         "instructions": "Is this detection correct, or is it a false alarm?",
         "choices": [{"value": "correct", "description": PERSON_DESC},
                     {"value": "false_alarm", "description": NOT_DESC}]}],
        "choice_q": "verdict", "not_values": ["false_alarm"],
        "scorers": {"choice": ("choice", "verdict", ["false_alarm"])}},
    "fa_pred": {"questions": [
        {"type": "predicate", "name": "false_alarm",
         "instructions": "This detection is a false alarm: the box does not mainly contain a real person. "
                         "It shows " + NOT_DESC + "."},
        {"type": "predicate", "name": "person",
         "instructions": "The box mainly contains a real person (pedestrian), even if small, blurry, "
                         "partly cut off or partly occluded."}],
        "choice_q": None, "not_values": [],
        "scorers": {"fa": ("pred_pos", "false_alarm"), "person": ("pred_neg", "person"),
                    "mean": ("mean", ["fa", "person"])}},
    "score5": {"questions": [
        {"type": "score", "name": "notperson",
         "instructions": "How likely is it that this detection is NOT a real person (a false alarm)?",
         "levels": [
             {"label": "definitely person", "description": "clearly a real person occupies the box"},
             {"label": "probably person", "description": "most likely a real person, but small, blurry or partly hidden"},
             {"label": "unsure", "description": "cannot tell whether it is a real person"},
             {"label": "probably not person", "description": "most likely background, an object or a fragment"},
             {"label": "definitely not person", "description": NOT_DESC}]}],
        "choice_q": None, "not_values": [],
        "scorers": {"score": ("score", "notperson", 4)}},
}


def call(path, qs, key, model):
    body = {"model": model,
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": CONTEXT},
                {"type": "input_image", "image_url": "data:image/jpeg;base64," + A.b64_file(path)}]}],
            "questions": qs}
    js, lat = A.post_json(URL, {"Authorization": f"Bearer {key}"}, body)
    return {a.get("name"): a for a in js.get("answers", [])}, lat, (js.get("usage") or {}).get("input_tokens"), js


def scores_of(ans, spec):
    out = {}
    for name, sc in spec["scorers"].items():
        kind = sc[0]
        try:
            if kind == "choice":
                a = ans[sc[1]]
                out[name] = sum(float(p["probability"]) for p in a["probabilities"] if p["value"] in sc[2])
            elif kind == "pred_pos":
                out[name] = float(ans[sc[1]]["probability"])
            elif kind == "pred_neg":
                out[name] = 1.0 - float(ans[sc[1]]["probability"])
            elif kind == "score":
                out[name] = float(ans[sc[1]]["score"]) / sc[2]
            elif kind == "mean":
                vals = [out[k] for k in sc[1] if k in out]
                out[name] = float(np.mean(vals)) if len(vals) == len(sc[1]) else None
        except (KeyError, TypeError, ValueError):
            out[name] = None
    return out


def score_of_row(r, scn):
    """CSV 행에서 점수: 점수 열은 s_<점수법> (예전 파일은 접두어 없음 → 그 열로 대체)"""
    return r.get("s_" + scn, r.get(scn))


def choice_of(ans, spec):
    q = spec["choice_q"]
    if q and q in ans and ans[q].get("type") == "choice":
        return ans[q]["choice"]
    return None


def run_set(sname, split, items, key, model, workers, limit):
    spec = SETS[sname]
    d = os.path.join(ROOT, "runs/api_sweep")
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, f"{split}_{sname}.csv")
    raw = os.path.join(d, f"{split}_{sname}_raw.jsonl")
    # choice = 실제 답(정확도용), s_<점수법> = "사람 아님" 점수. 점수법 이름 "choice"와 겹치지 않게 접두어를 붙인다
    fields = ["path", "label", "choice", "lat_ms", "in_tokens"] + ["s_" + k for k in spec["scorers"]]
    if os.path.exists(out):
        head = next(csv.reader(open(out)), [])
        if head != fields:
            sys.exit(f"[중단] {out} 의 열 구성이 현재 형식과 다릅니다: {head}")
    done = {r["path"] for r in csv.DictReader(open(out))} if os.path.exists(out) else set()
    todo = [it for it in items if it["path"] not in done]
    if limit:
        todo = todo[:max(0, limit - len(done))]
    print(f"  [{sname}/{split}] 완료 {len(done)}, 이번 {len(todo)} → {out}", flush=True)
    new = not os.path.exists(out)
    f, fr = open(out, "a", newline=""), open(raw, "a")
    w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()
    lock, n_ok, n_fail, t0 = threading.Lock(), 0, 0, time.time()

    def job(it):
        return it, call(it["path"], spec["questions"], key, model)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(job, it) for it in todo]
        for fu in as_completed(futs):
            try:
                it, (ans, lat, tok, js) = fu.result()
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                print(f"    실패: {str(e)[:200]}", flush=True)
                if n_fail >= 15:
                    print("    실패 15회 → 이 세트 중단", flush=True)
                    for x in futs:
                        x.cancel()
                    break
                continue
            sc = scores_of(ans, spec)
            with lock:
                w.writerow({"path": it["path"], "label": it["label"], "choice": choice_of(ans, spec) or "",
                            "lat_ms": f"{lat:.1f}", "in_tokens": "" if tok is None else tok,
                            **{"s_" + k: ("" if v is None else f"{v:.6f}") for k, v in sc.items()}})
                f.flush()
                fr.write(json.dumps({"path": it["path"], "response": js}, ensure_ascii=False) + "\n")
                fr.flush()
                n_ok += 1
                if n_ok % 50 == 0 or n_ok == len(todo):
                    print(f"    {n_ok}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    f.close()
    fr.close()


def evaluate(split, sname):
    p = os.path.join(ROOT, "runs/api_sweep", f"{split}_{sname}.csv")
    if not os.path.exists(p):
        return []
    rows = list(csv.DictReader(open(p)))
    res = []
    for scn in SETS[sname]["scorers"]:
        rr = [r for r in rows if score_of_row(r, scn) not in (None, "")]
        if len(rr) < 20:
            continue
        y = np.array([r["label"] == "not_person" for r in rr], int)
        s = np.array([float(score_of_row(r, scn)) for r in rr])
        if y.min() == y.max():
            continue
        r5, _, _ = C.removal_at(s, y, 0.05)
        r10, _, _ = C.removal_at(s, y, 0.10)
        res.append({"set": sname, "scorer": scn, "n": len(rr), "missing": len(rows) - len(rr),
                    "auroc": float(E.auroc(s, y)), "ece": float(E.ece(s, y)), "rm5": r5, "rm10": r10,
                    "uniq": int(len(np.unique(s))), "tokens": float(np.mean([float(r["in_tokens"]) for r in rr if r["in_tokens"]] or [0]))})
    return res


def val_table(sets):
    allr = [r for s in sets for r in evaluate("val", s)]
    allr.sort(key=lambda r: -r["auroc"])
    print("\n검증 영상 MOT17-10 (417장) — 문구 후보 순위 (AUROC 기준)")
    print(f"{'세트':<10}{'점수법':<9}{'n':>5}{'누락':>5}{'AUROC':>8}{'ECE':>8}{'제거@5%':>9}{'제거@10%':>10}{'고유값':>7}{'토큰':>7}")
    for r in allr:
        print(f"{r['set']:<10}{r['scorer']:<9}{r['n']:>5}{r['missing']:>5}{r['auroc']:>8.3f}{r['ece']:>8.3f}"
              f"{r['rm5']:>9.1%}{r['rm10']:>10.1%}{r['uniq']:>7}{r['tokens']:>7.0f}")
    print("참고: OmniJev 최종 헤드의 검증 AUROC 0.861 (15번 로그)")
    out = os.path.join(ROOT, "runs/api_sweep/val_summary.csv")
    if allr:
        w = csv.DictWriter(open(out, "w", newline=""), fieldnames=list(allr[0]))
        w.writeheader()
        w.writerows(allr)
        print(f"저장: {out}")
    return allr


def to_eval_csv(split, sname, scn, dst):
    """05_compare.py / 07 report 형식(label, p_not, choice, lat_ms, in_tokens)으로 변환"""
    rows = list(csv.DictReader(open(os.path.join(ROOT, "runs/api_sweep", f"{split}_{sname}.csv"))))
    nv = SETS[sname]["not_values"]
    w = csv.DictWriter(open(dst, "w", newline=""), fieldnames=["path", "label", "choice", "p_not", "lat_ms", "in_tokens"])
    w.writeheader()
    for r in rows:
        if score_of_row(r, scn) in (None, ""):
            continue
        s = float(score_of_row(r, scn))
        ch = ("not person" if r["choice"] in nv else "person") if r["choice"] else ("not person" if s > 0.5 else "person")
        w.writerow({"path": r["path"], "label": r["label"], "choice": ch, "p_not": s,
                    "lat_ms": r["lat_ms"], "in_tokens": r["in_tokens"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val"], default=None)
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    ap.add_argument("--final", action="store_true", help="검증 1등만 시험 영상에 1회")
    ap.add_argument("--force-test", action="store_true", help="시험 결과가 이미 있어도 다시 (권장하지 않음)")
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    os.chdir(ROOT)
    A.load_env(os.path.join(ROOT, ".env"))
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        sys.exit("[중단] .env 에 OPENAI_API_KEY 가 없습니다")
    print(f"시작 {time.strftime('%Y-%m-%d %H:%M:%S')} | 모델 {a.model}", flush=True)

    if a.split == "val":
        items = A.crop_list("val")
        for s in a.sets:
            print(f"\n[{s}] 검증 호출", flush=True)
            run_set(s, "val", items, key, a.model, a.workers, a.limit)
        val_table(a.sets)

    if a.final:
        allr = val_table(list(SETS))
        if not allr:
            sys.exit("[중단] 검증 결과가 없습니다. 먼저 --split val 실행")
        best = allr[0]
        print(f"\n검증 1등: 세트 {best['set']}, 점수법 {best['scorer']} (검증 AUROC {best['auroc']:.3f}) → 시험 영상 1회 채점", flush=True)
        dst_test = os.path.join(ROOT, "runs/eval/omnijev_openai_best.csv")
        if os.path.exists(os.path.join(ROOT, "runs/api_sweep", f"test_{best['set']}.csv")) and not a.force_test:
            print("  시험 결과가 이미 있어 다시 호출하지 않습니다(시험은 1회 원칙). 저장된 결과로 표만 만듭니다.")
        else:
            run_set(best["set"], "test", A.crop_list("test"), key, a.model, a.workers, 0)
        to_eval_csv("val", best["set"], best["scorer"], os.path.join(ROOT, "runs/eval/omnijev_val10_openai_best.csv"))
        to_eval_csv("test", best["set"], best["scorer"], dst_test)
        json.dump(best, open(os.path.join(ROOT, "runs/api_sweep/best.json"), "w"), ensure_ascii=False, indent=2)
        A.PROVIDERS["openai_best"] = {"tag": "openai_best", "usd_per_1m": 0.10}
        A.PROVIDERS["openai"]["tag"] = "openai_luna"
        A.report(["openai", "openai_best"])
    print(f"\n끝 {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
