import sys, os, json
sys.path.append(os.path.abspath("./Model_Modulization/Resume_Job_Weighted_Similarity_module/"))
import Recommendation_based_on_similarity as rbs

with open('E:/Project/12 노동고용/데이터/post_dataset.json', 'r', encoding='utf-8') as file:
    post_dataset = json.load(file)
with open('E:/Project/12 노동고용/데이터/resume_dataset.json', 'r', encoding='utf-8') as file:
    resume_dataset = json.load(file)


rbs.initialize()


for item in resume_dataset:
    results = rbs.pipeline(post_dataset , item['resume_description'])
    print('*'*10,'id: ',item['resume_id'],'*'*10)
    for result in results:
        print(f"ID: {result['id']}, 유사도: {result['similarity']}")

for item in post_dataset:
    results = rbs.pipeline(resume_dataset , item['jobPost_description'])
    print('*'*10,'id: ',item['jobPost_id'],'*'*10)
    for result in results:
        print(f"ID: {result['id']}, 유사도: {result['similarity']}")