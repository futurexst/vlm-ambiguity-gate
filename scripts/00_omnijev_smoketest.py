# =============================================================================
# test_first.py — OmniJev-0.8B 첫 판정 테스트
#
# 목적
#   1) 모델이 노트북(RTX 3060 6GB, WSL2)에서 정상 로딩되는지 확인
#   2) "사람 / 가려진 사람 / 오탐" 판정 확률이 상식적으로 나오는지 확인
#   3) 기준선 수치 기록: 로딩 시간, VRAM, 판정 1회 지연시간
#
# OmniJev 동작 원리 (한 줄 요약)
#   일반 VLM은 답을 글자(토큰) 단위로 한 개씩 이어 쓴다 → 답이 길수록 느리다.
#   OmniJev는 글로 답을 쓰지 않는다. 모델을 딱 한 번 통과시켜 "선택지마다 몇 %"인지만 받는다.
#   (서술형 답안이 아니라 객관식 OMR 채점이라고 보면 된다)
#   → 확률을 게이트 임계값(예: 오탐 확률 0.9 이상이면 박스 제거)에 바로 연결할 수 있다.
# =============================================================================

import time   # 시간 측정. perf_counter()는 해상도가 높아 지연시간(ms) 측정에 적합
import json   # 결과 딕셔너리를 보기 좋게 출력(json.dumps)
import torch  # GPU 동기화, VRAM 측정용

# MSO1 = OmniJev 모델을 불러오고 질문하는 도구.
# ~/work/OmniJev/mso/ 폴더 안에 있으므로, 반드시 ~/work/OmniJev 폴더에서 실행해야 한다.
# (파이썬은 import할 때 지금 있는 폴더부터 찾는다. 다른 폴더에서 실행하면 "No module named 'mso'" 에러)
from mso.infer import MSO1

# -----------------------------------------------------------------------------
# 경로 설정
# -----------------------------------------------------------------------------
# 판정 모델 = 백본(Qwen 원본) + OmniJev 부품(CKPT 폴더). 실행하려면 둘 다 필요하다.
#
#   부품                         | 파일                       | 하는 일 (비유)
#   ----------------------------|---------------------------|------------------------------------
#   백본 (Qwen이 만듦)            | qwen3.5-0.8b 폴더          | 이미지를 보고 문장을 이해함 (눈과 뇌)
#   LoRA 어댑터 (OmniJev)        | adapter_model.safetensors | 판정 요령을 적어 둔 메모장
#   판정 헤드 (OmniJev)          | head.pt                   | 선택지별 점수를 확률로 바꿈 (채점표)
#   단계 헤드 (OmniJev)          | ord.pt                    | score 유형 전용 채점표
#   보정값 (OmniJev)             | head_meta.json            | 질문 유형별 확률 보정값(온도)
#   특수 토큰 (OmniJev)          | new_tok_emb.pt            | 선택지 경계를 표시하는 토큰
#
# LoRA 쉽게: 백본은 "사진 설명은 잘하지만 객관식 시험 형식은 모르는 졸업생"이다.
#   제작자가 판정 문제 약 25만 개를 풀게 하면서, 원래 머릿속(원본 가중치)은 그대로 두고
#   옆에 붙인 작은 메모장에만 요령을 적게 했다. 이 메모장이 LoRA다. 떼면 원래 Qwen으로 돌아간다.
# LoRA의 r(=rank, 여기서는 32) = 메모장의 폭.
#   큰 행렬(예: 1000×1000 = 100만 개)을 직접 고치지 않고, 작은 행렬 두 개(1000×32, 32×1000)의 곱으로
#   "고칠 양"만 학습한다 → 6.4만 개(6.4%)만 학습. r이 크면 더 많이 바꿀 수 있지만 파일이 커진다.
# .pt = PyTorch로 저장한 파일이라는 확장자일 뿐이다. "모델"이라는 뜻은 아니다.
CKPT = "/home/uk24/work/models/omnijev-0.8b"   # OmniJev 부품 폴더(약 123MB)
BASE = "/home/uk24/work/models/qwen3.5-0.8b"   # 백본 폴더(약 1.6GB). 가중치는 고정(학습 안 함)

# 테스트 이미지 2장 (COCO val2017에서 받음. 로컬 테스트용, GitHub에는 올리지 않는다)
#   person.jpg : 사람이 있는 사진 → 정답 person
#   cats.jpg   : 사람이 없는 사진 → 정답 false positive
IMAGES = ["/home/uk24/work/test_images/person.jpg", "/home/uk24/work/test_images/person_crop0.jpg", "/home/uk24/work/test_images/cat_crop0.jpg", "/home/uk24/work/test_images/cat_crop1.jpg"]

# -----------------------------------------------------------------------------
# QUESTIONS: 질문 묶음
# -----------------------------------------------------------------------------
# 구조:  { 질문ID(문자열): 질문(딕셔너리), 질문ID: 질문, ... }
#   - 질문ID("what", "real")는 마음대로 정하는 이름표다. 결과도 같은 ID로 돌아온다.
#   - 여러 질문을 한 번에 넘기면 이미지는 한 번만 읽고 질문들이 그것을 공유한다.
#     → 질문을 1개에서 2개로 늘려도 비용이 2배가 되지 않는다.
#
# ─ 모든 질문에 공통인 항목 ──────────────────────────────────────────────
#   "type"         [필수] "choice" | "noul" | "score" 중 하나. 판정 방식을 정한다
#   "instructions" [필수] 질문 문장. 빠지면 KeyError.
#                         영어가 가장 잘 동작한다(제작자 안내).
#                         추가 맥락(트랙 정보 등)은 이 문장 앞에 붙여 넣는다.
#                         예: "Track 7, seen for 3 frames, detector score 0.41.\nWhat is shown in this image?"
#
# ─ type = "choice" : 여러 선택지 중 하나 고르기 → 채점: head.pt ──────────
#   선택지를 주는 방법은 아래 두 가지 중 하나 [둘 중 하나는 필수]
#     (A) "criteria": { 키: 설명, ... }   ← 이 코드에서 쓰는 방식
#           키   : 결과에 그대로 돌아오는 선택지 이름
#           설명 : [선택] 선택지의 뜻을 풀어 쓴 문장. None이나 ""도 된다.
#                  설명이 있으면 모델에는 "키: 설명" 형태로 들어가고, 없으면 키만 들어간다.
#                  → 설명을 구체적으로 쓰는 것이 판정 품질을 좌우한다(프롬프트 역할).
#     (B) "options": [ {"key": ..., "text": ...} 또는 {"key": ..., "region": {"box": [x1,y1,x2,y2]}}, ... ]
#           region은 이미지 안의 영역을 선택지로 쓰는 방식. 좌표는 0~1000으로 정규화한다.
#           → 크롭 대신 "원본 프레임 + YOLO 박스 여러 개 중 진짜 사람인 것은?"처럼 물어볼 수 있다.
#   선택지가 하나도 없으면 오류 대신 valid=False 결과가 돌아온다.
#   선택지끼리는 서로 못 보게 처리되므로 선택지 순서를 바꿔도 결과는 같다.
#   선택지 외에 "해당 없음(abstain)" 확률이 자동으로 함께 계산된다(따로 적지 않는다).
#
# ─ type = "noul" : 참/거짓 판정 (bool과 같은 뜻) → 채점: head.pt ──────────
#   "instructions"에 질문이 아니라 "판정할 문장(서술문)"을 쓴다.
#     좋은 예: "There is a real person in this image."
#     나쁜 예: "Is there a person?"  (동작은 하지만 학습 형식과 다르다)
#   "region": {"box": [...]}  [선택] 문장을 이미지의 특정 영역에 대해 판정할 때
#   선택지는 쓰지 않는다. 결과는 P(참) 하나.
#
# ─ type = "score" : 순서가 있는 단계 중 어디인지 → 채점: ord.pt (자동 전환) ─
#   (이 파일에서는 안 씀. 3단계 쓰러짐·이상 행동 긴급도용)
#   "levels": ["low", "medium", "high"]  [필수] 낮은 단계 → 높은 단계 순서로 적는다.
#   "보통 < 주의 < 위험"처럼 순서 자체가 의미라서 순서를 고려하는 전용 채점표(ord.pt)를 쓴다.
#   예: {"type": "score", "instructions": "How urgent is this situation?",
#        "levels": ["normal", "watch", "alert", "emergency"]}
# -----------------------------------------------------------------------------
QUESTIONS = {
    # 질문 1: 박스 안이 무엇인가 (1단계 오탐 필터의 핵심 질문)
    "what": {"type": "choice",
             "instructions": "What is shown in this image?",
             "criteria": {
                 # 키(결과에 돌아오는 이름)       설명(모델이 판단 기준으로 읽는 문장)
                 "person":          "a fully visible real person",
                 "occluded person": "a real person that is partly hidden",
                 # 오탐 유형을 구체적으로 나열할수록 마네킹·포스터 같은 경우를 잘 잡는다
                 "not person":      "not a real person: mannequin, poster, shadow, reflection or another object"}},

    # 질문 2: 같은 내용을 참/거짓으로 한 번 더 확인. what 결과와 비교해 일관성을 본다
    "real": {"type": "noul", "instructions": "There is a real person in this image."},
}

# -----------------------------------------------------------------------------
# 모델 로딩
# -----------------------------------------------------------------------------
t0 = time.perf_counter()

# MSO1(ckpt, base_model, tiny=False, max_pixels=768*28*28, align_options=True)
#   ckpt          [필수] OmniJev 부품 폴더
#   base_model    [필수] 백본 폴더
#   max_pixels    [선택] 이미지 최대 픽셀 수. 기본 약 60만 픽셀(768×28×28).
#                        이보다 큰 이미지는 줄여서 넣는다. 작게 하면 빨라지고 정확도는 떨어질 수 있다
#                        → 나중에 "해상도 vs 지연시간" 비교 실험 변수로 쓸 수 있다
#   align_options [선택] 선택지들이 같은 위치에서 시작하게 맞춤(순서 무관성). 기본값 유지
#   tiny          [선택] 테스트용 소형 모델. 쓰지 않는다
#
# 정밀도: GPU에서는 bfloat16(BF16)으로 자동 로딩된다. FP16이 아니다.
#
# branch 경로(자동):
#   Qwen3.5는 "빠른 요약형 층(선형 어텐션)"을 섞어 쓰는 구조다.
#   - 일반 어텐션: 매번 앞 내용을 전부 다시 훑어봄 → 정확하지만 길수록 느림
#   - 선형 어텐션: 앞 내용을 요약 메모 한 장에 덧쓰며 그 메모만 봄 → 빠름
#   요약 메모에는 모든 내용이 섞여서, 선택지끼리 못 보게 가리는 게 불가능하다.
#   그래서 이미지+질문을 한 번 읽어 둔 상태를 선택지 수만큼 복사해 따로따로 채점한다(갈림길 = branch).
# fla(flash-linear-attention) = 요약형 층을 GPU에서 빠르게 계산하는 전용 코드.
#   없으면 결과는 같고 속도만 느리다(실행 시 "falling back to reference implementation" 경고).
m = MSO1(CKPT, BASE)

# CPU는 GPU에 일을 맡기고 바로 다음 줄로 넘어간다(GPU는 뒤에서 따로 계산 = 비동기).
# 그래서 시간을 재기 전에 "GPU 계산이 다 끝날 때까지 기다리기"를 해야 실제 시간이 나온다.
# (식당 비유: 주문서를 넣자마자 스톱워치를 멈추면 요리 시간이 측정되지 않는다)
torch.cuda.synchronize()
# memory_allocated(): 지금 PyTorch가 실제로 차지한 VRAM(바이트) → GB로 변환
print(f"load: {time.perf_counter()-t0:.1f}s, VRAM after load: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

# -----------------------------------------------------------------------------
# 이미지별 판정 + 지연시간 측정
# -----------------------------------------------------------------------------
for img in IMAGES:
    # system_one(state, questions, packed=True)
    #   state     [필수] {"images": [이미지 경로, ...]}
    #               - "images" 리스트는 필수. 경로 문자열을 넣는다.
    #               - 2장 이상 넣으면 번호가 붙은 한 장짜리 패널로 이어 붙여 한 이미지로 넣는다.
    #                 → 질문에서 "the first picture / the second picture"로 가리킬 수 있다.
    #                 → 2단계 재식별("두 크롭이 같은 사람인가")은 이 방식으로 실험한다.
    #             "video" [선택] mso/video.py의 video_state()가 만들어 주는 값(16프레임 모자이크)
    #   questions [필수] 위의 QUESTIONS
    #   packed    [선택] 질문 여러 개를 한 번에 묶어서 처리할지(기본 True = 묶음, 빠름).
    #               False면 질문마다 따로 넣어서 질문 수만큼 느려진다.
    #               단, 0.8B는 branch 경로가 원래 묶어서 처리하므로 이 옵션이 무시된다.
    #               4B(일반 어텐션 백본)로 바꾸면 그때 의미가 생긴다.

    # 워밍업 = 준비운동. 아래 측정 호출과 똑같이 진짜로 모델에 넣지만, 결과와 시간은 버린다.
    #   첫 실행에는 GPU 계산 코드 준비, 메모리 확보 같은 "처음 한 번만 드는 비용"이 섞여 느리게 나오기 때문.
    m.system_one({"images": [img]}, QUESTIONS)  # warmup
    torch.cuda.synchronize()  # 워밍업의 남은 GPU 작업이 첫 측정에 섞이지 않게 비워 둔다

    # 측정 호출: 5회 반복해서 평균과 최솟값을 본다. ans에 담긴 마지막 결과를 출력한다
    times = []
    for _ in range(5):
        t = time.perf_counter()
        ans = m.system_one({"images": [img]}, QUESTIONS)
        torch.cuda.synchronize()                          # 판정 계산이 다 끝난 뒤 스톱워치를 멈춘다
        times.append((time.perf_counter() - t) * 1000)    # 초 → ms

    # ⚠️ 측정 주의: 모델은 사진을 넣기 전에 "손질"(파일 열기, 크기 줄이기, 숫자로 변환)을 한다.
    #    MSO1은 손질한 결과를 최근 4장까지 보관해서, 같은 사진이 또 오면 손질을 건너뛴다.
    #    이 테스트는 같은 사진을 반복해서 넣으므로 손질 시간이 빠져 실제보다 빠르게 나온다.
    #    실제 파이프라인은 매번 새 크롭이 들어오므로 그때 다시 측정한다.
    print(f"\n== {img.split('/')[-1]}  latency(2 questions): {sum(times)/len(times):.0f}ms avg, {min(times):.0f}ms min")

    # 결과 구조: { 질문ID: 답, ... }  (질문ID는 QUESTIONS의 키와 같다)
    #
    # choice 답:
    #   "choice"          가장 확률이 높은 선택지의 키 (자동 표시. 채택할지는 우리 게이트 규칙이 정한다)
    #   "probabilities"   선택지별 확률 (질문 유형별 온도로 보정된 값)
    #   "abstain"         "해당 없음" 확률. probabilities 합계 + abstain = 1
    #                     → 높으면 "주어진 선택지 중 맞는 게 없다"는 뜻. 사람 검토로 보낼 신호
    #   "valid"           판정이 정상적으로 만들어졌는지(선택지가 비어 있으면 False)
    #   "confidence"      확신도 = "찍은 것보다 얼마나 확신하는가"를 0~1로 바꾼 점수
    #                       공식: (K × 최고 확률 − 1) ÷ (K − 1)
    #                       K = 고를 수 있는 답의 개수 = 선택지 3개 + 해당 없음 1개 = 4 (캐시 4장과는 무관)
    #                       최고 확률 = probabilities 중 가장 큰 값
    #                       예: person 0.7293 → (4×0.7293 − 1) ÷ 3 = 1.9172 ÷ 3 = 0.639
    #                       0 = 찍은 수준(최고 확률이 1/K), 1 = 한 선택지에 100%
    #                     왜 필요한가: 2지선다의 60%와 5지선다의 60%는 의미가 다르다(찍기 50% vs 20%).
    #                       confidence로 바꾸면 선택지 개수가 달라도 기준 하나(예: 0.7 이상 자동 처리)로 쓸 수 있다.
    #                     출처: Jev 쪽에서 정한 정의를 OmniJev가 따른 것. 업계 표준 공식은 아니다.
    #                       (최고 확률(MSP)을 찍기 수준 1/K 기준으로 보정한 값. 코헨의 카파와 같은 구조)
    #                     → 1단계에서 MSP / Jev confidence / 마진(1등−2등) / 엔트로피 4가지를
    #                       라우팅 기준으로 MOT17에서 실측 비교해서 고른다.
    #   "latency_s"       전체 시간을 질문 수로 똑같이 나눈 값. 질문별로 따로 잰 시간이 아니다
    #   "latency_total_s" 이 호출 전체(전처리 + forward + 헤드) 시간(초)
    #
    # noul 답:
    #   "noul"            P(참). 0~1. 몇 이상을 "참"으로 볼지(0.5? 0.8?)는 우리가 정한다
    #
    # score 답(이 파일에서는 안 씀):
    #   "score" 가장 확률 높은 단계, "probabilities" 단계별 확률, "confidence" 확신도
    #
    # json.dumps 옵션:
    #   indent=2           한 줄로 몰아 쓰지 말고 줄을 바꿔 2칸씩 들여써서 보기 좋게 출력
    #   ensure_ascii=False 한글 같은 영어 외 문자를 \uXXXX 코드로 바꾸지 말고 그대로 출력
    print(json.dumps(ans, indent=2, ensure_ascii=False))

# 실행 중 가장 많이 썼던 순간의 VRAM. YOLO·CLIP-ReID를 같은 GPU에 올릴 여유를 판단하는 기준
print(f"\npeak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")