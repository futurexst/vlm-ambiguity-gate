"""질문 문구 후보 모음. 모든 후보는 'not person' 키를 가진다 (채점·학습 코드가 이 키를 기준으로 계산)."""
PROMPTS = {
    # P0: 지금까지 쓴 문구 (기준)
    "P0": {
        "what": {"type": "choice", "instructions": "What is shown in this image?",
                 "criteria": {"person": "a fully visible real person",
                              "occluded person": "a real person that is partly hidden",
                              "not person": "not a real person: mannequin, poster, shadow, reflection or another object"}},
        "real": {"type": "noul", "instructions": "There is a real person in this image."},
    },
    # P1: CCTV 맥락 명시 + 작거나 흐리거나 잘린 보행자도 사람으로 정의
    "P1": {
        "what": {"type": "choice",
                 "instructions": "This is a crop around a detected box from a surveillance camera. What does the box mainly contain?",
                 "criteria": {"person": "a real pedestrian, even if small, blurry or partly cut off",
                              "occluded person": "a real pedestrian that is partly hidden behind something",
                              "not person": "no real pedestrian: background, object, mannequin, poster, shadow or reflection"}},
        "real": {"type": "noul", "instructions": "This surveillance crop shows a real pedestrian, even if small or blurry."},
    },
    # P2: 선택지 2개로 단순화
    "P2": {
        "what": {"type": "choice",
                 "instructions": "This is a crop from a surveillance camera. What is in the center of the crop?",
                 "criteria": {"pedestrian": "a real person, even if small, blurry or partly visible",
                              "not person": "background, an object, a mannequin, a poster, a shadow or a reflection"}},
        "real": {"type": "noul", "instructions": "A real person stands or walks in the center of this crop."},
    },
    # P3: 가장 짧은 질문
    "P3": {
        "what": {"type": "choice", "instructions": "Is there a person in this image?",
                 "criteria": {"person": "yes, a person", "not person": "no person"}},
        "real": {"type": "noul", "instructions": "A person is visible in this image."},
    },
}
