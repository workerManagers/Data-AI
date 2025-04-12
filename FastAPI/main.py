from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import sys
import os
from typing import Literal
root_path= './FastAPI'
sys.path.append(os.path.abspath(f"{root_path}/"))
import Multi_Layer_MLP_module as model
import search_diagnosis_module as search

model.initialize(root_path)
search.initialize(root_path)

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
    try:
        result = model.pipeline(
            data.disease,
            data.sex,
            data.surgery,
            data.age,
            data.region
        )
        return PredictionOutput(predicted_value=float(result[0]))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search-diagnosis")
def search_diagnosis(keyword: str = ""):
    keyword = keyword.strip()
    if not keyword:
        return {"results": [], "message": "검색어가 비어있습니다."}
    diagnoses_list = search.search_diagnoses(keyword)
    if not diagnoses_list:
        return {"results": [], "message": f"'{keyword}'에 해당하는 병명이 없습니다."}
    return {"results": diagnoses_list}

class ResumeInput(BaseModel):
    resume: str

@app.post("/compare_resume")
def compare_resume(data: ResumeInput):
    return 0


# uvicorn FastAPI.main:app --reload
# http://127.0.0.1:8000/docs
