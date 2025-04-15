import logging
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import sys
import os
from typing import List, Literal

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

# 모델 및 모듈 초기화
logger.info("모델(Multi_Layer_MLP_module) 초기화 시작")
model.initialize(root_path)
logger.info("모델(Multi_Layer_MLP_module) 초기화 완료")

logger.info("진단명 검색 모듈(search_diagnosis_module) 초기화 시작")
search.initialize(root_path)
logger.info("진단명 검색 모듈(search_diagnosis_module) 초기화 완료")

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

@app.get("/health")
def health_check():
    return {"status": "ok"}

# uvicorn main:app --host=0.0.0.0 --port=8000
# http://127.0.0.1:8000/docs
