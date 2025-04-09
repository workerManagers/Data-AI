from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import joblib

app = FastAPI(title="요양일 예측 API")

# ✅ 입력값 스키마 정의
class PredictRequest(BaseModel):
    disease: str
    gender: str
    surgery: str
    age_group: str
    region: str

# ✅ 모델 정의 및 로딩
def load_model_and_encoders():
    global model, encoders

    class MLPRegressor(nn.Module):
        def __init__(self, input_dims, embed_dim=8):
            super().__init__()
            self.embeds = nn.ModuleList([
                nn.Embedding(dim, embed_dim) for dim in input_dims
            ])
            self.fc1 = nn.Linear(embed_dim * len(input_dims), 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.fc2 = nn.Linear(128, 64)
            self.out = nn.Linear(64, 1)

        def forward(self, x):
            x = torch.cat([emb(x[:, i]) for i, emb in enumerate(self.embeds)], dim=1)
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.fc2(x))
            return self.out(x).squeeze()

    # 인코더 로딩
    encoders = {
        '병명': joblib.load('encoder_disease.pkl'),
        '성별': joblib.load('encoder_gender.pkl'),
        '수술여부': joblib.load('encoder_surgery.pkl'),
        '연령대': joblib.load('encoder_age.pkl'),
        '지역본부': joblib.load('encoder_region.pkl'),
    }

    input_dims = [len(encoders[col].classes_) for col in encoders]
    model = MLPRegressor(input_dims)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

# ✅ API 서버 시작 시 모델 로드
@app.on_event("startup")
def startup_event():
    load_model_and_encoders()

# ✅ 예측 API 라우트
@app.post("/predict")
def predict(request: PredictRequest):
    input_dict = {
        '병명': [request.disease],
        '성별': [request.gender],
        '수술여부': [request.surgery],
        '연령대': [request.age_group],
        '지역본부': [request.region]
    }
    input_df = pd.DataFrame(input_dict)

    for col in input_df.columns:
        input_df[col] = encoders[col].transform(input_df[col])

    input_tensor = torch.tensor(input_df.values, dtype=torch.long)

    with torch.no_grad():
        pred = model(input_tensor)
        return {"predicted_care_duration": round(pred.item(), 2)}
