from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.abspath("./Model_Modulization/Multi_Layer_MLP_module/"))
from Multi_Layer_MLP_module import initialize, pipeline

# FastAPI 앱 초기화
app = FastAPI()

# 초기화 (모델, 인코더 등)
initialize()

# 입력 데이터 형식 정의 (요청용)
from typing import Literal
from pydantic import BaseModel

class PredictRequest(BaseModel):
    disease: str  # 여기는 값이 너무 많으니까 그냥 str로 유지 추천
    sex: Literal["남자", "여자"]
    surgery: Literal["예", "아니오"]
    age: Literal["30세미만", "30-39세", "40-49세", "50-59세", "60세이상"]
    region: Literal["서울지역", "부산지역", "대구지역", "경인지역", "광주지역", "대전지역"]



# API 라우터
@app.post("/predict")
def predict(data: PredictRequest):
    output = pipeline(data.disease, data.sex, data.surgery, data.age, data.region)
    return {
        "rescaled_bottleneck_output": float(output[0])
    }
