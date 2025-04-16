import re
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

def initialize():
    global model
    model = SentenceTransformer("intfloat/multilingual-e5-small")

# ✅ 블록 단위 분할
def split_into_token_safe_chunks(text, token_limit=384):
    text = text.replace('\n', ' ')
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())

    tokenizer = model.tokenizer
    chunks, current_chunk = [], ""

    for sent in sentences:
        temp_chunk = (current_chunk + " " + sent).strip()
        token_len = len(tokenizer.encode(temp_chunk, add_special_tokens=True))

        if token_len <= token_limit:
            current_chunk = temp_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sent

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# ✅ 문장 유사도 (벡터화 + 코사인 유사도)
def weighted_average_similarity(resume, job_description):
    resume_chunks = split_into_token_safe_chunks(resume)
    job_chunks = split_into_token_safe_chunks(job_description)

    if not resume_chunks or not job_chunks:
        return 0.0

    res_vecs = model.encode(resume_chunks, convert_to_tensor=True)
    job_vecs = model.encode(job_chunks, convert_to_tensor=True)

    # [res_n, job_n] 형태로 pairwise 코사인 유사도 계산
    sim_matrix = F.cosine_similarity(
        res_vecs.unsqueeze(1),  # [N, 1, D]
        job_vecs.unsqueeze(0),  # [1, M, D]
        dim=-1
    )  # [N, M]

    # flatten 후 가중합
    sims = sim_matrix.flatten()
    weights = sims ** 2
    weighted_sum = torch.sum(sims * weights)
    total_weight = torch.sum(weights)

    return (weighted_sum / total_weight).item() if total_weight > 0 else 0.0

# ✅ 전체 유사도 파이프라인
def pipeline(dataset, input_str):
    if not dataset:
        raise ValueError("입력 데이터셋이 비어 있습니다.")
    if not input_str.strip():
        raise ValueError("입력 문장이 비어 있습니다.")

    input_str = input_str.strip().replace('\n', ' ')

    # 데이터 형식 결정
    keys = set(dataset[0].keys())
    if keys == {'jobPost_id', 'jobPost_description'}:
        description_list = [{"id": d["jobPost_id"], "text": d["jobPost_description"]} for d in dataset]
    elif keys == {'resume_id', 'resume_description'}:
        description_list = [{"id": d["resume_id"], "text": d["resume_description"]} for d in dataset]
    else:
        raise ValueError("지원되지 않는 데이터 포맷입니다.")

    # 입력은 한 번만 임베딩
    results = []
    for entry in description_list:
        sim = weighted_average_similarity(input_str, entry["text"])
        results.append({"id": entry["id"], "similarity": sim})

    top_sorted = sorted(results, key=lambda x: x["similarity"], reverse=True)
    return [{"id": r["id"], "similarity": f"{round(r['similarity'] * 100, 2)}점"} for r in top_sorted]
