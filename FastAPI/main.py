@@ -1,18 +1,36 @@
 import logging
 from fastapi import FastAPI, Query, HTTPException
 from pydantic import BaseModel
 import sys
 import os
 from typing import List, Dict, Any, Literal, Union
 root_path= '.'
 
 # 로깅 설정
 logging.basicConfig(
     level=logging.INFO,
     format="%(asctime)s - %(levelname)s - %(message)s"
 )
 logger = logging.getLogger(__name__)
 
 root_path = '.'
 sys.path.append(os.path.abspath(f"{root_path}/"))
 
 import Multi_Layer_MLP_module as model
 import search_diagnosis_module as search
 import Recommendation_based_on_similarity as similar
 
 # 모델 및 모듈 초기화
 logger.info("모델(Multi_Layer_MLP_module) 초기화 시작")
 model.initialize(root_path)
 logger.info("모델(Multi_Layer_MLP_module) 초기화 완료")
 
 logger.info("진단명 검색 모듈(search_diagnosis_module) 초기화 시작")
 search.initialize(root_path)
 logger.info("진단명 검색 모듈(search_diagnosis_module) 초기화 완료")
 
 logger.info("유사도 추천 모듈(Recommendation_based_on_similarity) 초기화 시작")
 similar.initialize()
 logger.info("유사도 추천 모듈(Recommendation_based_on_similarity) 초기화 완료")
 
 app = FastAPI()
 
 @@ -28,6 +46,7 @@ class PredictionOutput(BaseModel):
 
 @app.post("/predict", response_model=PredictionOutput)
 async def predict_care_duration(data: PredictionInput):
     logger.info(f"/predict 호출: {data.dict()}")
     try:
         result = model.pipeline(
             data.disease,
 @@ -36,23 +55,29 @@ async def predict_care_duration(data: PredictionInput):
             data.age,
             data.region
         )
         logger.info(f"/predict 결과: {result}")
         return PredictionOutput(predicted_value=float(result[0]))
     except ValueError as ve:
         logger.warning(f"/predict ValueError: {ve}")
         raise HTTPException(status_code=400, detail=str(ve))
     except Exception as e:
         logger.error(f"/predict Exception: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=str(e))
 
 @app.get("/search-diagnosis")
 def search_diagnosis(keyword: str = ""):
     logger.info(f"/search-diagnosis 호출: keyword='{keyword}'")
     keyword = keyword.strip()
     if not keyword:
         logger.info("/search-diagnosis: 검색어가 비어있음")
         return {"results": [], "message": "검색어가 비어있습니다."}
     diagnoses_list = search.search_diagnoses(keyword)
     if not diagnoses_list:
         logger.info(f"/search-diagnosis: '{keyword}'에 해당하는 병명 없음")
         return {"results": [], "message": f"'{keyword}'에 해당하는 병명이 없습니다."}
     logger.info(f"/search-diagnosis 결과: {diagnoses_list}")
     return {"results": diagnoses_list}
 
 
 class SimilarityResult(BaseModel):
     id: Union[str, int]
     similarity: str
 @@ -63,14 +88,14 @@ class CompareRequest(BaseModel):
 
 @app.post("/compare", response_model=List[SimilarityResult])
 def compare_resume(data: CompareRequest):
     logger.info(f"/compare 호출: input_text 길이={len(data.input_text)}, dataset 크기={len(data.dataset)}")
     try:
         return similar.pipeline(data.dataset, data.input_text)
         result = similar.pipeline(data.dataset, data.input_text)
         logger.info(f"/compare 결과: {result}")
         return result
     except Exception as e:
         logger.error(f"/compare Exception: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"서버 내부 오류: {str(e)}")
 
 @app.get("/health")
 async def health_check():
     return {"status": "healthy"}
 
 # uvicorn FastAPI.main:app --reload
 # http://127.0.0.1:8000/docs
