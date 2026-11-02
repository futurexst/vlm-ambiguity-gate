#!/usr/bin/env python3
"""
07_eval_api.py — 판단 모델 API(OpenAI Decisions · TypeSafe Jev 1.14)를 OmniJev와 같은 조건으로 채점·비교

같은 조건
  - 같은 크롭: OmniJev 최종 채점 CSV의 path를 그대로 사용 (I2 크롭, 검증 10번 417장 / 시험 02번 400장)
  - 같은 질문: prompts.py의 P1 (what = 선택지 3개, real = 참/거짓)
  - 같은 지표: 정확도, AUROC, ECE, 제거@사람손실 5%/10% (03·05번 함수 재사용)
  - 같은 합격선 규칙: 검증 영상의 진짜 사람 p_not 90%/95% 분위수 → 시험 영상에 1회 적용

흐름
  1) 사전 점검: 업체마다 검증 크롭 2장만 호출해 응답 형식 확인 (실패한 업체는 건너뜀)
  2) 본 호출: 검증 → 시험. 크롭마다 즉시 CSV에 저장하므로, 끊겨도 다시 실행하면 이어서 함
  3) 비교표 출력 + runs/eval/api_compare_summary.json 저장

사용법
  python scripts/07_eval_api.py                       # .env에 키가 있는 업체 전부
  python scripts/07_eval_api.py --providers openai    # 하나만
  python scripts/07_eval_api.py --limit 3             # 업체·영상마다 3장만 (동작 확인용)
  python scripts/07_eval_api.py --report-only         # 호출 없이 저장된 결과로 표만

API 키: 저장소 루트 .env (절대 커밋 금지)
  OPENAI_API_KEY=...
  TYPESAFE_API_KEY=...
"""
import argparse
import base64
import csv
import importlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import PROMPTS  # noqa: E402

E = importlib.import_module("03_eval_omnijev")   # auroc, ece (기존 로그와 같은 정의)
C = importlib.import_module("05_compare")        # removal_at

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = {"val": "runs/eval/omnijev_val10_headft_i2p1.csv",   # 크롭 목록 출처 (OmniJev 최종 채점과 같은 이미지)
       "test": "runs/eval/omnijev_headft_i2p1.csv"}
PREFIX = {"val": "val10_", "test": ""}                    # 기존 파일 이름 규칙: omnijev_<prefix><tag>.csv

PROVIDERS = {
    "openai": {"url": "https://api.openai.com/v1/decisions", "env": "OPENAI_API_KEY",
               "tag": "openai_luna", "usd_per_1m": 0.10},
    "jev": {"url": "https://api.typesafe.ai/v1/systemone", "env": "TYPESAFE_API_KEY",
            "tag": "jev114", "usd_per_1m": 0.042},
}
OMNI = [("OmniJev 제로샷 (P1·I2)", "zeroshot_i2p1"), ("OmniJev 최종 (헤드 미세조정)", "headft_i2p1")]


# ----------------------------------------------------------------------------- 공통
def load_env(path):
    """.env의 KEY=VALUE를 환경변수로 (이미 있는 값은 유지). 키 값은 화면에 출력하지 않는다"""
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def post_json(url, headers, body, timeout=90, tries=6):
    """POST + 재시도(429·5xx·네트워크 오류는 2,4,8..최대 60초 대기). 반환: (응답 JSON, 성공한 호출의 지연 ms)"""
    data = json.dumps(body).encode()
    last = None
    for i in range(tries):
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={**headers, "Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                js = json.loads(r.read())
            return js, (time.perf_counter() - t0) * 1000
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:600]
            last = f"HTTP {e.code}: {msg}"
            if e.code in (408, 409, 429, 500, 502, 503, 504) and i < tries - 1:
                time.sleep(min(60, 2 ** (i + 1)))
                continue
            raise RuntimeError(last)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = f"네트워크 오류: {e}"
            if i < tries - 1:
                time.sleep(min(60, 2 ** (i + 1)))
                continue
            raise RuntimeError(last)
    raise RuntimeError(last)


def b64_file(path):
    return base64.b64encode(open(path, "rb").read()).decode("ascii")


def b64_padded(path, lo=280, hi=1568):
    """Jev 이미지 제한(가로·세로 280~1568px, 총 78,400px 이상)에 맞춤.
    큰 쪽이 1568을 넘으면 비율 유지 축소, 작은 쪽은 크롭을 늘리지 않고 회색 여백만 붙인다(사람 부분 내용은 그대로)."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if max(w, h) > hi:
        s = hi / max(w, h)
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
        w, h = im.size
    W, H = max(w, lo), max(h, lo)
    if (W, H) != (w, h):
        canvas = Image.new("RGB", (W, H), (114, 114, 114))
        canvas.paste(im, ((W - w) // 2, (H - h) // 2))
        im = canvas
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ----------------------------------------------------------------------------- 업체별 요청·응답
def openai_call(path, q, key, model):
    w, r = q["what"], q["real"]
    body = {
        "model": model,
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": "Evaluate this image crop."},
            {"type": "input_image", "image_url": "data:image/jpeg;base64," + b64_file(path)}]}],
        "questions": [
            {"type": "choice", "name": "what", "instructions": w["instructions"],
             "choices": [{"value": k, "description": v} for k, v in w["criteria"].items()]},
            {"type": "predicate", "name": "real", "instructions": r["instructions"]}],
    }
    js, lat = post_json(PROVIDERS["openai"]["url"], {"Authorization": f"Bearer {key}"}, body)
    ans = {a.get("name"): a for a in js.get("answers", [])}
    a = ans.get("what") or {}
    if a.get("type") != "choice":
        raise RuntimeError(f"what 답 없음 또는 거부: {json.dumps(a)[:300]}")
    probs = {p["value"]: float(p["probability"]) for p in a["probabilities"]}
    p_real = (ans.get("real") or {}).get("probability")
    tok = (js.get("usage") or {}).get("input_tokens")
    return a["choice"], probs, p_real, tok, lat, js


def jev_call(path, q, key, model):
    w, r = q["what"], q["real"]
    ref = "img-crop-0001"
    body = {
        "model": model,
        "state": {"crop": ref},
        "assets": {ref: {"type": "image", "format": "jpg", "data_base64": b64_padded(path)}},
        "questions": {
            "what": {"type": "choice", "instructions": "Look at `crop`. " + w["instructions"],
                     "criteria": w["criteria"]},
            "real": {"type": "noul", "instructions": "Look at `crop`. " + r["instructions"]}},
    }
    hdr = {"Authorization": f"Bearer {key}", "TypeSafe-Danger-Zone-Images": "true"}
    js, lat = post_json(PROVIDERS["jev"]["url"], hdr, body)
    ans = js.get("answers") or {}
    a = ans.get("what") or {}
    if a.get("type") != "choice" or "probabilities" not in a:
        raise RuntimeError(f"what 답 형식이 다름: {json.dumps(a)[:300]}")
    probs = {k: float(v) for k, v in a["probabilities"].items()}
    p_real = (ans.get("real") or {}).get("noul")
    tok = (js.get("usage") or {}).get("input_tokens")
    return a["choice"], probs, p_real, tok, lat, js


CALL = {"openai": openai_call, "jev": jev_call}
FIELDS = ["path", "label", "source", "choice", "p_person", "p_occluded", "p_not", "p_real", "lat_ms", "in_tokens"]


# ----------------------------------------------------------------------------- 실행
def crop_list(split):
    p = os.path.join(ROOT, REF[split])
    rows = list(csv.DictReader(open(p)))
    if not rows or "path" not in rows[0] or "label" not in rows[0]:
        sys.exit(f"[중단] {p} 에 path/label 열이 없습니다. 열: {list(rows[0]) if rows else '없음'}")
    out = []
    for r in rows:
        path = os.path.expanduser(r["path"])
        if not os.path.isabs(path):
            path = os.path.join(ROOT, path)
        if not os.path.exists(path):
            sys.exit(f"[중단] 크롭 이미지가 없습니다: {path}")
        out.append({"path": path, "label": r["label"], "source": r.get("source", "")})
    return out


def run_split(name, split, items, q, key, model, limit):
    tag = PROVIDERS[name]["tag"]
    out_csv = os.path.join(ROOT, "runs/eval", f"omnijev_{PREFIX[split]}{tag}.csv")
    raw = os.path.join(ROOT, "runs/api_raw", f"{tag}_{split}.jsonl")
    err = os.path.join(ROOT, "runs/api_raw", f"{tag}_{split}_errors.txt")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(raw), exist_ok=True)
    done = set()
    if os.path.exists(out_csv):
        done = {r["path"] for r in csv.DictReader(open(out_csv))}
    todo = [it for it in items if it["path"] not in done]
    if limit:
        todo = todo[:max(0, limit - len(done))]
    print(f"  [{name}/{split}] 전체 {len(items)}, 이미 완료 {len(done)}, 이번에 호출 {len(todo)} → {out_csv}", flush=True)
    new = not os.path.exists(out_csv)
    fcsv = open(out_csv, "a", newline="")
    wr = csv.DictWriter(fcsv, fieldnames=FIELDS)
    if new:
        wr.writeheader()
    fraw, ferr = open(raw, "a"), open(err, "a")
    fails, t0 = 0, time.time()
    for i, it in enumerate(todo, 1):
        try:
            choice, probs, p_real, tok, lat, js = CALL[name](it["path"], q, key, model)
        except Exception as e:  # noqa: BLE001 — 한 장 실패는 기록하고 다음 장으로
            fails += 1
            ferr.write(f"{it['path']}\t{e}\n")
            ferr.flush()
            print(f"    실패 {fails}회 연속: {os.path.basename(it['path'])} — {str(e)[:200]}", flush=True)
            if fails >= 10:
                print(f"  [{name}/{split}] 10회 연속 실패 → 이 업체 중단", flush=True)
                return False
            continue
        fails = 0
        wr.writerow({"path": it["path"], "label": it["label"], "source": it["source"], "choice": choice,
                     "p_person": probs.get("person", ""), "p_occluded": probs.get("occluded person", ""),
                     "p_not": probs.get("not person", 0.0), "p_real": "" if p_real is None else p_real,
                     "lat_ms": f"{lat:.1f}", "in_tokens": "" if tok is None else tok})
        fcsv.flush()
        fraw.write(json.dumps({"path": it["path"], "response": js}, ensure_ascii=False) + "\n")
        fraw.flush()
        if i % 25 == 0 or i == len(todo):
            print(f"    {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    return True


def preflight(name, items, q, key, model):
    print(f"\n[사전 점검] {name}: 검증 크롭 2장", flush=True)
    for it in items[:2]:
        try:
            choice, probs, p_real, tok, lat, js = CALL[name](it["path"], q, key, model)
        except Exception as e:  # noqa: BLE001
            print(f"  실패 → {name} 건너뜀: {str(e)[:400]}", flush=True)
            return False
        print(f"  {os.path.basename(it['path'])} 정답={it['label']} → choice={choice} "
              f"p_not={probs.get('not person')} p_real={p_real} tokens={tok} {lat:.0f}ms", flush=True)
        if "not person" not in probs:
            print(f"  'not person' 확률이 응답에 없음 → {name} 건너뜀. 응답: {json.dumps(js)[:400]}", flush=True)
            return False
    return True


# ----------------------------------------------------------------------------- 비교표
def load(tag, split):
    p = os.path.join(ROOT, "runs/eval", f"omnijev_{PREFIX[split]}{tag}.csv")
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return None
    d = {"y": np.array([r["label"] == "not_person" for r in rows], int),
         "s": np.array([float(r["p_not"]) for r in rows]),
         "pred": np.array([r["choice"] == "not person" for r in rows], int),
         "lat": np.array([float(r["lat_ms"]) for r in rows]) if "lat_ms" in rows[0] else None,
         "tok": [float(r["in_tokens"]) for r in rows if r.get("in_tokens") not in (None, "")]}
    return d


def metrics(d):
    y, s, pred = d["y"], d["s"], d["pred"]
    r5, _, _ = C.removal_at(s, y, 0.05)
    r10, _, _ = C.removal_at(s, y, 0.10)
    return {"n": int(len(y)), "acc": float((pred == y).mean()), "person_acc": float((pred[y == 0] == 0).mean()),
            "auroc": float(E.auroc(s, y)), "ece": float(E.ece(s, y)), "rm5": r5, "rm10": r10,
            "lat_med": float(np.median(d["lat"])) if d["lat"] is not None else None,
            "lat_p90": float(np.percentile(d["lat"], 90)) if d["lat"] is not None else None}


def op_point(val, test, q):
    t = float(np.quantile(val["s"][val["y"] == 0], q))
    return {"thr": t, "val_loss": float((val["s"][val["y"] == 0] > t).mean()),
            "test_rm": float((test["s"][test["y"] == 1] > t).mean()),
            "test_loss": float((test["s"][test["y"] == 0] > t).mean())}


def report(names):
    models = list(OMNI) + [(f"{n} API ({PROVIDERS[n]['tag']})", PROVIDERS[n]["tag"]) for n in names]
    summary = {}
    print("\n" + "=" * 100)
    print("시험 영상 MOT17-02 (같은 400장, 입력 I2)")
    print(f"{'모델':<34}{'n':>5}{'정확도':>8}{'사람정답':>9}{'AUROC':>8}{'ECE':>8}"
          f"{'제거@5%':>9}{'제거@10%':>10}{'지연중앙':>10}{'지연p90':>9}")
    for label, tag in models:
        te = load(tag, "test")
        if te is None:
            print(f"{label:<34}  (결과 없음)")
            continue
        m = metrics(te)
        summary[tag] = {"test": m}
        lat = f"{m['lat_med']:>8.0f}ms{m['lat_p90']:>7.0f}ms" if m["lat_med"] is not None else ""
        print(f"{label:<34}{m['n']:>5}{m['acc']:>8.3f}{m['person_acc']:>9.3f}{m['auroc']:>8.3f}{m['ece']:>8.3f}"
              f"{m['rm5']:>9.1%}{m['rm10']:>10.1%}{lat}")
    print("\n운영 합격선: 검증 영상(MOT17-10) 진짜 사람 p_not 분위수로 정하고 시험 영상에 1회 적용")
    print(f"{'모델':<34}{'분위수':>7}{'합격선':>9}{'검증손실':>9}{'시험제거':>9}{'시험손실':>9}")
    for label, tag in models:
        va, te = load(tag, "val"), load(tag, "test")
        if va is None or te is None:
            print(f"{label:<34}  (검증 또는 시험 결과 없음)")
            continue
        summary.setdefault(tag, {})["val"] = metrics(va)
        for q in (0.90, 0.95):
            o = op_point(va, te, q)
            summary[tag][f"op_q{int(q * 100)}"] = o
            print(f"{label:<34}{int(q * 100):>6}%{o['thr']:>9.4f}{o['val_loss']:>9.1%}{o['test_rm']:>9.1%}{o['test_loss']:>9.1%}")
    print("\nAPI 사용량 (검증+시험, 응답의 input_tokens 기준)")
    for n in names:
        tag = PROVIDERS[n]["tag"]
        toks = [t for s in ("val", "test") for t in ((load(tag, s) or {}).get("tok") or [])]
        if toks:
            usd = sum(toks) / 1e6 * PROVIDERS[n]["usd_per_1m"]
            print(f"  {n:<8} 호출 {len(toks)}회, 평균 {np.mean(toks):.0f} 토큰/장, 합계 {sum(toks):,.0f} 토큰, "
                  f"추정 비용 ${usd:.4f} (입력 100만 토큰당 ${PROVIDERS[n]['usd_per_1m']} 기준)")
            summary.setdefault(tag, {})["tokens"] = {"calls": len(toks), "total": float(sum(toks)), "usd_est": usd}
        else:
            print(f"  {n:<8} 토큰 정보 없음")
    print("\n주의: OmniJev 지연은 로컬 RTX 3060, API 지연은 네트워크 왕복 포함. Jev 이미지는 제작사 기준 실험(preview) 기능.")
    out = os.path.join(ROOT, "runs/eval/api_compare_summary.json")
    json.dump(summary, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"요약 저장: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", nargs="+", default=list(PROVIDERS), choices=list(PROVIDERS))
    ap.add_argument("--prompt", default="P1")
    ap.add_argument("--openai-model", default="gpt-6-luna")
    ap.add_argument("--jev-model", default="jev-1.14.0")
    ap.add_argument("--limit", type=int, default=0, help="업체·영상마다 최대 장수 (0 = 전부)")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()

    os.chdir(ROOT)
    load_env(os.path.join(ROOT, ".env"))
    q = PROMPTS[a.prompt]
    model = {"openai": a.openai_model, "jev": a.jev_model}
    print(f"시작 {time.strftime('%Y-%m-%d %H:%M:%S')} | 질문 {a.prompt} | 업체 {a.providers}", flush=True)

    if not a.report_only:
        items = {s: crop_list(s) for s in ("val", "test")}
        sizes = [Image.open(it["path"]).size for it in items["test"][:50]]
        print(f"크롭: 검증 {len(items['val'])}장, 시험 {len(items['test'])}장 | 시험 앞 50장 크기 "
              f"가로 {min(w for w, _ in sizes)}~{max(w for w, _ in sizes)}, 세로 {min(h for _, h in sizes)}~{max(h for _, h in sizes)}px",
              flush=True)
        for name in a.providers:
            key = os.environ.get(PROVIDERS[name]["env"], "")
            if not key:
                print(f"\n[{name}] {PROVIDERS[name]['env']} 가 .env에 없어 건너뜀", flush=True)
                continue
            if not preflight(name, items["val"], q, key, model[name]):
                continue
            for split in ("val", "test"):
                print(f"\n[{name}] {split} 호출 ({model[name]})", flush=True)
                if not run_split(name, split, items[split], q, key, model[name], a.limit):
                    break
    report(a.providers)
    print(f"끝 {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
