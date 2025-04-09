# ✅ 필요 패키지
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

# ✅ 전역 모델/인코더 로딩 (처음 한 번만)
    # 메모리 줄이기 -> 모델 정의 안에 넣기 
def load_model_and_encoders():
    global model, encoders

    # 모델 정의
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

# ✅ 예측 함수
def predict_care_duration(disease, gender, surgery, age_group, region):

    input_dict = {
        '병명': [disease],
        '성별': [gender],
        '수술여부': [surgery],
        '연령대': [age_group],
        '지역본부': [region]
    }
    input_df = pd.DataFrame(input_dict)

    for col in input_df.columns:
        input_df[col] = encoders[col].transform(input_df[col])

    input_tensor = torch.tensor(input_df.values, dtype=torch.long)

    with torch.no_grad():
        pred = model(input_tensor)
        return pred.item()

# ✅ 딸깍하면 바로 실행되는 부분
if __name__ == "__main__":
    # 예시 입력값
    disease = 'S000(머리덮개의 얕은 손상)'
    gender = '성별_남자'
    surgery = '수술여부_예'
    age_group = '연령대_30-39세'
    region = '지역본부_서울지역'
    load_model_and_encoders()
    # 예측 실행
    predicted_value = predict_care_duration(disease, gender, surgery, age_group, region)
    print(f"📌 예측된 요양일: {predicted_value:.2f}일")
