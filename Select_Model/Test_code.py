import sys, os, json
sys.path.append(os.path.abspath("./Lightweight_Models/"))
# import intfloat_multilingual_e5_small as module
# import all_MiniLM_L6_v2 as module
# import jhgan_ko_sbert_sts as module
# import bongsoo_albert_small_kor_sbert_v1 as module
import smartmind_roberta_ko_small_tsdae as module

with open('E:/Project/12 노동고용/데이터/post_dataset.json', 'r', encoding='utf-8') as file:
    post_dataset = json.load(file)
with open('E:/Project/12 노동고용/데이터/resume_dataset.json', 'r', encoding='utf-8') as file:
    resume_dataset = json.load(file)


module.initialize()


for item in resume_dataset:
    results = module.pipeline(post_dataset , item['resume_description'])
    print('*'*10,'id: ',item['resume_id'],'*'*10)
    for result in results:
        print(f"ID: {result['id']}, 유사도: {result['similarity']}")

for item in post_dataset:
    results = module.pipeline(resume_dataset , item['jobPost_description'])
    print('*'*10,'id: ',item['jobPost_id'],'*'*10)
    for result in results:
        print(f"ID: {result['id']}, 유사도: {result['similarity']}")