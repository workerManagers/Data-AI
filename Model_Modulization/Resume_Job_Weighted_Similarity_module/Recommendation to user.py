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
def weighted_average_similarity(resume, jobPost_description):
    resume_chunks = split_into_token_safe_chunks(resume)
    job_chunks = split_into_token_safe_chunks(jobPost_description)

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
    print("\n🔍 자소서 vs 채용공고 유사도 비교 시작")

    # ✅ 자소서 직접 입력
    self_intro = """저는 꾸준한 노력과 협업을 통해 문제를 해결해나가는 소프트웨어 엔지니어가 되고자 합니다. 대학교 재학 중 다양한 팀 프로젝트를 통해 실전 감각을 익혔으며, 특히 캡스톤디자인에서는 딥러닝 기반 이미지 분류기를 개발하여 3개월간의 협업을 성공적으로 마무리했습니다. 이 과정에서 TensorFlow와 Python을 활용한 모델 학습 및 성능 개선 경험을 쌓았습니다.\n또한 교내 알고리즘 스터디를 통해 자료구조와 알고리즘에 대한 이해를 심화하였으며, C++과 Java를 활용해 다양한 문제를 해결해보았습니다. 이외에도 클라우드 기반 환경에서 웹 애플리케이션을 개발한 경험이 있어, AWS 환경에서 EC2, RDS 등을 사용해 백엔드 서버를 구축한 바 있습니다.\n저는 항상 팀의 목표를 최우선으로 생각하며 소통하고, 새로운 기술을 두려워하지 않고 배우는 것을 즐깁니다. 삼성전자의 글로벌 환경에서 끊임없이 성장하며, 더 나은 사용자 경험을 제공하는 소프트웨어를 개발하고 싶습니다."""

    # ✅ JSON 파일에서 채용공고 로딩 (id 포함)
    try:
        json_path = "/Users/hwangtaeeon/Documents/GitHub/Data-AI/Model_Modulization/Resume_Job_Weighted_Similarity_module/post_dataset.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {e}")
        exit(1)

    jobPost_description_list = [
        {"jobPost_id": item.get("jobPost_id", ), "text": item.get("jobPost_description", "")}
        for idx, item in enumerate(data)
        if item.get("jobPost_description")
    ]

    results = []

    for job in jobPost_description_list:
        jobPost_id = job["jobPost_id"]
        job_text = job["text"]
        sim = weighted_average_similarity(self_intro, job_text)

        job_text = job_text.strip().replace('\n', ' ')
        self_intro = self_intro.strip().replace('\n', ' ')

        results.append({
            "jobPost_id": jobPost_id,
            "similarity": sim,
        })

    # 🔽 유사도 순 정렬
    sorted_results = sorted(results, key=lambda x: x["similarity"], reverse=True)

    # ✅ JSON 형식으로 상위 결과 추출
    top_json_results = [
        {"jobPost_id": item["jobPost_id"], "similarity": round(item["similarity"], 4)}
        for item in sorted_results
    ]

    # ✅ JSON 저장
    output_path = "top_match_results.json"
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(top_json_results, f, ensure_ascii=False, indent=4)
        print(f"\n✅ 결과가 '{output_path}' 파일에 저장되었습니다.")
    except Exception as e:
        print(f"❌ 결과 저장 실패: {e}")
