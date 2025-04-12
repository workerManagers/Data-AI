from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import sys
import os
from typing import Literal

sys.path.append(os.path.abspath("./Model_Modulization/Multi_Layer_MLP_module/"))
from Multi_Layer_MLP_module import initialize, pipeline, search_diagnoses

act = 'softplus'
initialize(act)

app = FastAPI()

class PredictionOutput(BaseModel):
    predicted_value: float

# 예측 API (쿼리 파라미터 방식)
@app.get("/predict", response_model=PredictionOutput)
async def predict_care_duration(
    disease: str = Query(..., description="병명"),
    sex: Literal["남자", "여자"] = Query(..., description="성별"),
    surgery: Literal["예", "아니오"] = Query(..., description="수술 여부"),
    age: Literal["30세미만", "30-39세", "40-49세", "50-59세", "60세이상"] = Query(..., description="연령대"),
    region: Literal["부산지역", "대구지역", "광주지역", "서울지역", "경인지역", "대전지역"] = Query(..., description="지역")
):
    try:
        result = pipeline(disease, sex, surgery, age, region)
        return PredictionOutput(predicted_value=float(result[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search-diagnosis")
def search_diagnosis(keyword: str = ""):
    keyword = keyword.strip()
    if not keyword:
        return {"results": [], "message": "검색어가 비어있습니다."}
    diagnoses_list = search_diagnoses(keyword)
    if not diagnoses_list:
        return {"results": [], "message": f"'{keyword}'에 해당하는 병명이 없습니다."}
    return {"results": diagnoses_list}

# uvicorn Model_Modulization.Multi_Layer_MLP_module.main:app --reload
# http://127.0.0.1:8000/docs
