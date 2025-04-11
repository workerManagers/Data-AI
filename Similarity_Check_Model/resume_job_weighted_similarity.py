# -*- coding: utf-8 -*-
import torch
import re
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer

# ✅ 모델 및 토크나이저 로딩
model = SentenceTransformer('jhgan/ko-sbert-sts')
tokenizer = AutoTokenizer.from_pretrained('jhgan/ko-sbert-sts')

# ✅ 문장 기준 + 토큰 수 기준 블록 쪼개기
def split_into_token_safe_chunks(text, token_limit=512, char_hint=200):
    text = text.replace('\n', ' ')
    
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    results = []

    current_chunk = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        temp_chunk = (current_chunk + " " + sent).strip()
        token_len = len(tokenizer.encode(temp_chunk, add_special_tokens=True))

        if token_len <= token_limit and len(temp_chunk) <= char_hint * 2:
            current_chunk = temp_chunk
        else:
            if current_chunk:
                results.append(current_chunk.strip())
            current_chunk = sent  # 새 문장 시작

    if current_chunk:
        results.append(current_chunk.strip())

    return results

# ✅ 평균 임베딩 방식
def encode_and_average(text):
    chunks = split_into_token_safe_chunks(text)
    embeddings = [model.encode(chunk, convert_to_tensor=True) for chunk in chunks]
    return torch.stack(embeddings).mean(dim=0)

# ✅ 가중 평균 방식
def weighted_average_similarity(resume, job_posting):
    resume_chunks = split_into_token_safe_chunks(resume)
    job_chunks = split_into_token_safe_chunks(job_posting)

    sims = []
    for res in resume_chunks:
        res_vec = model.encode(res, convert_to_tensor=True)
        for job in job_chunks:
            job_vec = model.encode(job, convert_to_tensor=True)
            sim = util.cos_sim(res_vec, job_vec).item()
            sims.append((sim, sim**2))

    if not sims:
        return 0.0

    weighted_sum = sum(sim * weight for sim, weight in sims)
    total_weight = sum(weight for _, weight in sims)
    return weighted_sum / total_weight if total_weight != 0 else 0.0

# ✅ 디버깅용: 블록별 유사도 확인용
def weighted_average_similarity_debug(resume, job_posting):
    resume_chunks = split_into_token_safe_chunks(resume)
    job_chunks = split_into_token_safe_chunks(job_posting)

    debug_info = []  # (resume_chunk, job_chunk, sim, weight)
    sims = []

    for res in resume_chunks:
        res_vec = model.encode(res, convert_to_tensor=True)
        for job in job_chunks:
            job_vec = model.encode(job, convert_to_tensor=True)
            sim = util.cos_sim(res_vec, job_vec).item()
            weight = sim ** 2
            sims.append((sim, weight))
            debug_info.append((res, job, sim, weight))

    if not sims:
        return 0.0, []

    weighted_sum = sum(sim * weight for sim, weight in sims)
    total_weight = sum(weight for _, weight in sims)
    weighted_avg = weighted_sum / total_weight if total_weight != 0 else 0.0

    return weighted_avg, debug_info

# ✅ 유사도 계산 통합 함수
def calculate_similarity(resume, job_posting, mode='mean'):
    if mode == 'mean':
        resume_vec = encode_and_average(resume)
        job_vec = encode_and_average(job_posting)
        return util.cos_sim(resume_vec, job_vec).item()
    elif mode == 'weighted':
        return weighted_average_similarity(resume, job_posting)
    else:
        raise ValueError("mode는 'mean' 또는 'weighted'만 가능합니다.")

# ✅ 예시 입력
resume = """
[클린시티 - 건물 청소 및 환경미화 인력 채용] 클린시티는 오피스, 상가, 학교 등 다양한 시설의 위생관리 서비스를 제공하는 전문 환경미화 기업입니다. 본 직무는 정해진 구역의 실내·외 청소 및 폐기물 수거, 시설 미화 등의 작업을 수행하며, 반복성과 체력 소모가 높은 직무입니다. [주요 업무] - 사무실, 계단, 복도, 화장실 등 지정 구역 청소 - 분리수거, 쓰레기 수거 및 폐기물 배출 - 청소도구 및 장비 관리, 청소용품 재고 점검 - 정기 소독 및 방역 작업 보조 (교육 이수 후) [자격 요건] - 학력 무관, 연령 무관 (고령자 환영) - 청결·위생 의식 보유자 - 반복 작업 및 실내외 이동 업무 수행 가능자 - 기본적인 장비 사용 및 업무 보고 가능자 [우대 사항] - 청소·미화 경력자 - 감염병 예방교육, 위생안전교육 이수자 - 장애인, 고령자, 경력단절자 우대 작고 묵묵한 일이지만, 모두가 깨끗하게 살아갈 수 있는 기본을 만들어갈 분들의 많은 지원 바랍니다.
"""

job_posting = """
저는 누가 보지 않아도 성실히 자기 구역을 책임지는 청소 인력이 되고 싶습니다. 그동안 학교 청소 보조, 공동주택 계단 미화 등의 일을 하며 먼지 하나, 얼룩 하나까지 신경 쓰는 습관을 들였고, 정해진 루틴에 따라 철저히 청소하면서 ‘위생’의 중요성을 체감했습니다. 작업 중에는 무릎과 허리를 반복해서 사용해야 하는 일이 많기에 무리하지 않고 자세를 바르게 유지하는 법을 익혔고, 장갑과 마스크를 항상 착용하며 감염성 오염물도 안전하게 다루는 법을 배웠습니다. 매일같이 같은 일을 반복해도, 저는 그 일이 누군가에게는 쾌적한 하루의 시작이 된다고 믿고 있습니다. 클린시티에서 묵묵히, 그러나 정확하게 자기 자리를 지키는 환경미화원이 되고 싶습니다.
"""

# ✅ 실행
mean_score = calculate_similarity(resume, job_posting, mode='mean')
weighted_score, debug_info = weighted_average_similarity_debug(resume, job_posting)

print(f"[평균 기반 유사도] {mean_score:.4f}")
print(f"[가중 평균 기반 유사도] {weighted_score:.4f}\n")

# ✅ 디버깅 정보 출력
for i, (res, job, sim, weight) in enumerate(debug_info):
    print(f"🔹 Pair {i+1}")
    print(f"   - 이력서 블록: {res}")
    print(f"   - 채용공고 블록: {job}")
    print(f"   - 유사도: {sim:.4f}")
    print(f"   - 가중치(sim^2): {weight:.4f}\n")
#0.5204, 0.4680 # 관련 없는 
#0.63, 0.59 # 관련 있는 거