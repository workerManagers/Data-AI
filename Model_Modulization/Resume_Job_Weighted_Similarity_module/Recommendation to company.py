# -*- coding: utf-8 -*-
import json
import torch
import re
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer

# ✅ 모델 및 토크나이저 로딩
model = SentenceTransformer('jhgan/ko-sbert-sts')
tokenizer = AutoTokenizer.from_pretrained('jhgan/ko-sbert-sts')

# ✅ 문장 + 토큰 수 기준 블록 쪼개기
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
            current_chunk = sent

    if current_chunk:
        results.append(current_chunk.strip())

    return results

# ✅ 유사도 계산 함수
def weighted_average_similarity(resume, resume_description):
    resume_chunks = split_into_token_safe_chunks(resume)
    job_chunks = split_into_token_safe_chunks(resume_description)

    sims = []
    for res in resume_chunks:
        res_vec = model.encode(res, convert_to_tensor=True)
        for job in job_chunks:
            job_vec = model.encode(job, convert_to_tensor=True)
            sim = util.cos_sim(res_vec, job_vec).item()
            weight = sim ** 2
            sims.append((sim, weight))

    if not sims:
        return 0.0

    weighted_sum = sum(sim * weight for sim, weight in sims)
    total_weight = sum(weight for _, weight in sims)
    return weighted_sum / total_weight if total_weight != 0 else 0.0

# ✅ 메인 실행
if __name__ == "__main__":
    print("\n🔍 채용공고 vs 자소서 유사도 비교 시작")

    # ✅ 채용공고 직접 입력 
    jobpost_input = """[삼성전자 - DX부문 소프트웨어 엔지니어 신입 채용]\n삼성전자는 글로벌 IT 시장을 선도할 인재를 모집합니다. 당사는 차세대 소프트웨어 기술을 기반으로 한 혁신적인 제품과 서비스를 통해 고객의 삶을 더욱 풍요롭게 하고자 합니다. DX부문은 스마트폰, 가전제품, 헬스케어 등 다양한 분야에서 소프트웨어 개발 및 서비스 플랫폼을 구축하고 있으며, 다음과 같은 인재를 찾고 있습니다.\n\n[주요 업무]\n- 임베디드 시스템 및 어플리케이션 소프트웨어 개발\n- 대용량 데이터 처리 및 클라우드 기반 서비스 개발\n- AI/ML 알고리즘을 활용한 기능 고도화\n- 글로벌 서비스 운영을 위한 시스템 아키텍처 설계 및 개발\n\n[자격 요건]\n- 컴퓨터공학, 전자공학 등 관련 전공 학사 이상\n- Python, C/C++, Java 등 프로그래밍 언어 활용 능력\n- 자료구조, 알고리즘, OS, 네트워크 등 기본 CS 지식 보유\n- 협업 및 커뮤니케이션 능력 우수자\n\n[우대 사항]\n- 오픈소스 프로젝트 참여 경험\n- AI/ML, 빅데이터 관련 프로젝트 수행 경험\n- 영어 및 기타 외국어 커뮤니케이션 능력\n\n혁신을 이끌어갈 도전적인 인재들의 많은 지원 바랍니다."""

    # ✅ JSON 파일에서 자소서 로딩 (id 포함)
    try:
        json_path = "/Users/hwangtaeeon/Documents/GitHub/Data-AI/Model_Modulization/Resume_Job_Weighted_Similarity_module/resume_dataset.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {e}")
        exit(1)

    resume_description_list = [
        {"resume_id": item.get("resume_id", ), "text": item.get("resume_description", "")}
        for idx, item in enumerate(data)
        if item.get("resume_description")
    ]

    results = []

    for job in resume_description_list:
        resume_id = job["resume_id"]
        job_text = job["text"]
        sim = weighted_average_similarity(jobpost_input, job_text)

        job_text = job_text.strip().replace('\n', ' ')
        jobpost_input = jobpost_input.strip().replace('\n', ' ')

        results.append({
            "resume_id": resume_id,
            "similarity": sim,
        })

    # 🔽 유사도 순 정렬
    sorted_results = sorted(results, key=lambda x: x["similarity"], reverse=True)

    # ✅ JSON 형식으로 상위 결과 추출
    top_json_results = [
        {"resume_id": item["resume_id"], "similarity": round(item["similarity"], 4)}
        for item in sorted_results
    ]

    # ✅ JSON 저장
    output_path = "top_resume_results.json"
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(top_json_results, f, ensure_ascii=False, indent=4)
        print(f"\n✅ 결과가 '{output_path}' 파일에 저장되었습니다.")
    except Exception as e:
        print(f"❌ 결과 저장 실패: {e}")
