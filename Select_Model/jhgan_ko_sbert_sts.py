import re
from sentence_transformers import SentenceTransformer, util

def initialize():
    global model
    model = SentenceTransformer('jhgan/ko-sbert-sts')

def split_into_token_safe_chunks(text, token_limit=512, char_hint=200):
    text = text.replace('\n', ' ')
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())

    results = []
    current_chunk = ""
    tokenizer = model.tokenizer

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

def pipeline(dataset, input_str):
    if not dataset:
        raise ValueError("입력 데이터셋이 비어 있습니다.")
    if not input_str.strip():
        raise ValueError("입력 문장이 비어 있습니다.")

    input_str = input_str.strip().replace('\n', ' ')

    keys = set(dataset[0].keys())
    if keys == {'jobPost_id', 'jobPost_description'}:
        description_list = [
            {"id": item.get("jobPost_id", ""), "text": item.get("jobPost_description", "")}
            for item in dataset
        ]
    elif keys == {'resume_id', 'resume_description'}:
        description_list = [
            {"id": item.get("resume_id", ""), "text": item.get("resume_description", "")}
            for item in dataset
        ]
    else:
        raise ValueError("지원되지 않는 데이터 포맷입니다.")

    results = []
    for entry in description_list:
        item_id = entry["id"]
        text = entry["text"].strip().replace('\n', ' ')
        sim = weighted_average_similarity(input_str, text)
        results.append({
            "id": item_id,
            "similarity": sim,
        })

    sorted_results = sorted(results, key=lambda x: x["similarity"], reverse=True)

    top_json_results = [
        {"id": item["id"], "similarity": f'{round(item["similarity"]*100, 2)}점'}
        for item in sorted_results
    ]
    return top_json_results
