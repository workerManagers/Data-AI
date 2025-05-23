import logging
from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
from typing import List, Dict, Any, Literal, Union
import uuid

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

class PredictionInput(BaseModel):
    disease: str
    sex: Literal["남자", "여자"]
    surgery: Literal["예", "아니오"]
    age: Literal["30세미만", "30-39세", "40-49세", "50-59세", "60세이상"]
    region: Literal["부산지역", "대구지역", "광주지역", "서울지역", "경인지역", "대전지역"]

class PredictionOutput(BaseModel):
    predicted_value: float

@app.post("/predict", response_model=PredictionOutput)
async def predict_care_duration(data: PredictionInput):
    logger.info(f"/predict 호출: {data.dict()}")
    try:
        result = model.pipeline(
            data.disease,
            data.sex,
            data.surgery,
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

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 결과를 저장할 딕셔너리
matching_results = {}

class SimilarityResult(BaseModel):
    id: Union[str, int]
    similarity: str

class CompareRequest(BaseModel):
    input_text: str
    dataset: List[Dict[str, Any]]

# 백그라운드에서 실행될 매칭 함수
async def process_matching(input_text: str, dataset: List[Dict[str, Any]], task_id: str):
    try:
        # 기존 AI 매칭 로직 실행
        result = similar.pipeline(dataset, input_text)
        matching_results[task_id] = {"status": "completed", "result": result}
        logger.info(f"Task {task_id} completed successfully")
    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
        matching_results[task_id] = {"status": "failed", "error": str(e)}

@app.post("/compare")
async def compare_resume(data: CompareRequest, background_tasks: BackgroundTasks):
    logger.info(f"/compare 호출: input_text 길이={len(data.input_text)}, dataset 크기={len(data.dataset)}")
    try:
        task_id = str(uuid.uuid4())
        matching_results[task_id] = {"status": "processing"}
        
        # 백그라운드 작업으로 처리
        background_tasks.add_task(
            process_matching,
            data.input_text,
            data.dataset,
            task_id
        )
        
        # 즉시 작업 ID 반환
        return JSONResponse({
            "task_id": task_id,
            "status": "processing"
        })
    except Exception as e:
        logger.error(f"/compare Exception: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {str(e)}")

@app.get("/compare/status/{task_id}")
async def check_status(task_id: str):
    result = matching_results.get(task_id, {"status": "not_found"})
    return JSONResponse(result)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# uvicorn FastAPI.main:app --reload
# http://127.0.0.1:8000/docs
