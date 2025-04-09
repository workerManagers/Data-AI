import pandas as pd
import itertools
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import joblib
import json

plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows용 한글 폰트
plt.rcParams['axes.unicode_minus'] = False     # 음수 부호 깨짐 방지
matplotlib.use('Agg')

################################################################################################
####################################### Path #############################################
################################################################################################

file_name = 'MLP'

model_save_dir = f"./Care_Duration_Prediction_Model/Modeling/{file_name}/Model"
result_save_dir = f"./Care_Duration_Prediction_Model/Modeling/{file_name}/Result"
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(result_save_dir, exist_ok=True)

model_path = os.path.join(model_save_dir, "best_model.pt")
result_path = os.path.join(result_save_dir, "predict_result.png")

################################################################################################
####################################### Data load #############################################
################################################################################################
df = pd.read_csv('./Care_Duration_Prediction_Model/Data_Preprocessing/Regular_Expression/final_data.csv')
print(df.describe())
print(df.info())
print(df.columns)
print(df.head())
print('*'*50)
print('*'*50)
print()
################################################################################################
####################################### Create data set ######################################
################################################################################################
genders = ['성별_남자', '성별_여자']
surgeries = ['수술여부_예', '수술여부_아니오']
ages = ['연령대_30세미만', '연령대_30-39세', '연령대_40-49세', '연령대_50-59세', '연령대_60세이상']
regions = ['지역본부_서울지역', '지역본부_부산지역', '지역본부_대구지역', '지역본부_경인지역', '지역본부_광주지역', '지역본부_대전지역']

combinations = list(itertools.product(genders, surgeries, ages, regions))

data_set = []

for _, row in df.iterrows():
    disease = row['병명']
    for gender, surgery, age, region in combinations:
        input_row = [disease, gender, surgery, age, region]
        output_row = [
            row.get(gender, None),
            row.get(surgery, None),
            row.get(age, None),
            row.get(region, None)
        ]
        data_set.append(input_row + output_row)
data_df = pd.DataFrame(data_set, columns=['병명', '성별', '수술여부', '연령대', '지역본부','성별_요양일', '수술여부_요양일', '연령대_요양일', '지역본부_요양일'])
print(data_df.head())
print('*'*50)
print('*'*50)
print()
################################################################################################
####################################### Data Preprocessing ######################################
################################################################################################

# 결측치(NaN)가 있는 행 제거
data_df = data_df.dropna().reset_index(drop=True)
print(data_df.isna().sum())
# 복사본 유지
processed_df = data_df.copy()

# Label Encoding: 병명
le = LabelEncoder()
processed_df['병명'] = le.fit_transform(processed_df['병명'])
preprocessor_dir = "./FastAPI/Multi_Layer_MLP_LabelEncoder.pkl"
joblib.dump(le, preprocessor_dir)

# One-Hot Encoding: 성별, 수술여부, 연령대, 지역본부
processed_df = processed_df.replace({True: 1, False: 0})
categorical_cols = ['성별', '수술여부', '연령대', '지역본부']
processed_df = pd.get_dummies(processed_df, columns=categorical_cols)

# 정규화: 출력값 4개
scaler = StandardScaler()
output_cols = ['성별_요양일', '수술여부_요양일', '연령대_요양일', '지역본부_요양일']
processed_df[output_cols] = scaler.fit_transform(processed_df[output_cols])
preprocessor_dir ="./FastAPI/Multi_Layer_MLP_StandardScaler.pkl"
joblib.dump(scaler, preprocessor_dir)


# 전처리 후의 입력 데이터프레임
X_columns = processed_df.drop(columns=output_cols).columns.tolist()

# 저장
with open('./FastAPI/Multi_Layer_MLP_input_columns.json', 'w', encoding='utf-8') as f:
    json.dump(X_columns, f, ensure_ascii=False, indent=2)


# 최종 입력과 출력 분리
X = processed_df.drop(columns=output_cols)
y = processed_df[output_cols]

print(X.head())
print(y.head())

# Train/valid/test split (60/20/20 비율 기준)
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.1, stratify=X['병명'], random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, stratify=X_trainval['병명'], random_state=42)

print(f'X_train_shape: {X_train.shape}')
print(f'X_val_shape: {X_val.shape}')
print(f'X_test_shape: {X_test.shape}')
print(f'y_train_shape: {y_train.shape}')
print(f'y_val_shape: {y_val.shape}')
print(f'y_test_shape: {y_test.shape}')
print('*'*50)
print('*'*50)
print()

# 병명 인덱스만 따로 추출 (임베딩 입력용)
disease_train = torch.tensor(X_train['병명'].values, dtype=torch.long)
disease_val = torch.tensor(X_val['병명'].values, dtype=torch.long)
disease_test = torch.tensor(X_test['병명'].values, dtype=torch.long)

# 병명 외 나머지 feature 추출
X_train_tensor = torch.tensor(X_train.drop(columns=['병명']).astype(np.float32).values)
X_val_tensor = torch.tensor(X_val.drop(columns=['병명']).astype(np.float32).values)
X_test_tensor = torch.tensor(X_test.drop(columns=['병명']).astype(np.float32).values)

y_train_tensor = torch.tensor(y_train.astype(np.float32).values)
y_val_tensor = torch.tensor(y_val.astype(np.float32).values)
y_test_tensor = torch.tensor(y_test.astype(np.float32).values)

# TensorDataset: (병명ID, 입력특성, 타겟)
train_dataset = TensorDataset(disease_train, X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(disease_val, X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(disease_test, X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)
test_loader = DataLoader(test_dataset, batch_size=128)

################################################################################################
####################################### AI Design and Training #################################
################################################################################################

class ComplexMLPWithEmbedding(nn.Module):
    def __init__(self, num_diseases, embedding_dim, input_dim, hidden_dims, bottleneck_dim, output_dim):
        super(ComplexMLPWithEmbedding, self).__init__()

        self.embedding = nn.Embedding(num_diseases, embedding_dim)

        layers = []
        total_input_dim = embedding_dim + input_dim  # 병명 임베딩 + 나머지 입력
        prev_dim = total_input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(0.3))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, bottleneck_dim))
        # layers.append(nn.Sigmoid())
        layers.append(nn.Softplus())
        layers.append(nn.Linear(bottleneck_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, disease_ids, features):
        embedded = self.embedding(disease_ids)  # (batch, embedding_dim)
        x = torch.cat((embedded, features), dim=1)
        return self.network(x)

# 하이퍼파라미터 정의
num_diseases = X['병명'].nunique()
embedding_dim = 16
input_dim = X_train_tensor.shape[1]  # 병명 제외 나머지 입력
hidden_dims = [256, 128, 128, 64, 64]
bottleneck_dim = 1
output_dim = 4

model = ComplexMLPWithEmbedding(num_diseases, embedding_dim, input_dim, hidden_dims, bottleneck_dim, output_dim)

# Loss 및 Optimizer 정의
#criterion = nn.MSELoss()
#criterion = nn.L1Loss()
criterion = nn.SmoothL1Loss()  # Huber
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 학습 기록용 변수
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience = 30
counter = 0

# for epoch in range(1000):
#     model.train()
#     epoch_train_loss = 0.0
#     for disease_id, xb, yb in train_loader:
#         disease_id, xb, yb = disease_id.to(device), xb.to(device), yb.to(device)
#         optimizer.zero_grad()
#         preds = model(disease_id, xb)
#         loss = criterion(preds, yb)
#         loss.backward()
#         optimizer.step()
#         epoch_train_loss += loss.item() * xb.size(0)
#     epoch_train_loss /= len(train_loader.dataset)
#     train_losses.append(epoch_train_loss)
#
#     # validation
#     model.eval()
#     epoch_val_loss = 0.0
#     with torch.no_grad():
#         for disease_id, xb, yb in val_loader:
#             disease_id, xb, yb = disease_id.to(device), xb.to(device), yb.to(device)
#             preds = model(disease_id, xb)
#             loss = criterion(preds, yb)
#             epoch_val_loss += loss.item() * xb.size(0)
#     epoch_val_loss /= len(val_loader.dataset)
#     val_losses.append(epoch_val_loss)
#
#     # ReduceLROnPlateau 확인
#     prev_lr = optimizer.param_groups[0]['lr']
#     scheduler.step(epoch_val_loss)
#     new_lr = optimizer.param_groups[0]['lr']
#     if new_lr < prev_lr:
#         print(f"📉 Epoch {epoch + 1:03d} | LR reduced from {prev_lr:.8f} to {new_lr:.8f}")
#
#     print(f"Epoch {epoch+1:03d} | Train Loss: {epoch_train_loss:.8f} | Val Loss: {epoch_val_loss:.8f}")
#
#     # Early stopping
#     if epoch_val_loss < best_val_loss:
#         best_val_loss = epoch_val_loss
#         counter = 0
#         torch.save(model.state_dict(), "best_model.pt")
#     else:
#         counter += 1
#         if counter >= patience:
#             print("🛑 Early stopping triggered.")
#             break
#
# torch.save(model.state_dict(), model_path)

################################################################################################
####################################### AI Test ################################################
################################################################################################

model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
model.eval()

preds_list = []
targets_list = []

with torch.no_grad():
    for disease_id, xb, yb in test_loader:
        disease_id, xb = disease_id.to(device), xb.to(device)
        preds = model(disease_id, xb).cpu().numpy()
        preds_list.append(preds)
        targets_list.append(yb.numpy())

preds_all = np.vstack(preds_list)
targets_all = np.vstack(targets_list)

preds_original = scaler.inverse_transform(preds_all)
targets_original = scaler.inverse_transform(targets_all)

################################################################################################
####################################### Save ###################################################
################################################################################################


# 시각화: 예측 vs 실제 (산점도)
plt.figure(figsize=(12, 8))
for i, label in enumerate(['성별_요양일', '수술여부_요양일', '연령대_요양일', '지역본부_요양일']):
    plt.subplot(2, 2, i+1)
    plt.plot(targets_original[:, i], color = 'black')
    plt.plot(preds_original[:, i],color = 'red' )
    plt.ylabel(f"value {label}")
    plt.title(label)
plt.tight_layout()
plt.savefig(result_path)
plt.close()

print(f"\n📊 Evaluation Metrics (단위: 원래 스케일)\n{'=' * 40}")

for i, label in enumerate(['성별_요양일', '수술여부_요양일', '연령대_요양일', '지역본부_요양일']):
    y_true = targets_original[:, i]
    y_pred = preds_original[:, i]

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # %

    print(f"📌 [{label}]")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  MAPE : {mape:.2f}%\n")


################################################################################################
####################################### Final task #############################################
################################################################################################

# hook 출력 저장용 리스트
bottleneck_outputs = []

# hook 함수 정의
def save_bottleneck_output(module, input, output):
    bottleneck_outputs.append(output.detach().cpu().numpy())

# bottleneck ReLU 레이어에 hook 등록
bottleneck_layer = model.network[21]
hook_handle = bottleneck_layer.register_forward_hook(save_bottleneck_output)

# 전체 데이터셋 (train + val + test)
full_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
full_loader = DataLoader(full_dataset, batch_size=128, shuffle=False)

# 추론 (forward만 하면 hook이 자동 작동)
with torch.no_grad():
    for disease_id, xb, _ in full_loader:
        disease_id, xb = disease_id.to(device), xb.to(device)
        _ = model(disease_id, xb)  # preds는 필요 없음, hook이 bottleneck 저장함

# 병합
bottleneck_all = np.vstack(bottleneck_outputs)  # shape: (전체 샘플 수, 1)

# hook 해제 (메모리 누수 방지)
hook_handle.remove()

################################################################################################
####################################### Save ###################################################
################################################################################################

# 성별, 연령대, 지역본부 복원
def reverse_one_hot(df, prefix):
    cols = [col for col in df.columns if col.startswith(prefix + '_')]
    return df[cols].idxmax(axis=1).str.replace(f"{prefix}_", "")

full_diseases = []
full_sex = []
full_age = []
full_region = []
full_y_original = []
for i in range(len(full_dataset)):
    disease_id, features, y_target = full_dataset[i]
    full_diseases.append(le.inverse_transform([disease_id.item()])[0])
    feature_np = features.numpy()
    feature_df = pd.DataFrame([feature_np], columns=X.columns.drop("병명"))
    full_sex.append(reverse_one_hot(feature_df, "성별")[0])
    full_age.append(reverse_one_hot(feature_df, "연령대")[0])
    full_region.append(reverse_one_hot(feature_df, "지역본부")[0])
    full_y_original.append(scaler.inverse_transform(y_target.view(1, -1).numpy())[0])
full_y_original = np.array(full_y_original)


result_df = pd.DataFrame({
    "병명": full_diseases,
    "성별": full_sex,
    "연령대": full_age,
    "지역본부": full_region,
    "성별_요양일": full_y_original[:, 0],
    "수술여부_요양일": full_y_original[:, 1],
    "연령대_요양일": full_y_original[:, 2],
    "지역본부_요양일": full_y_original[:, 3],
    "bottleneck_output": bottleneck_all.flatten()
})

# 저장
final_save_dir = f"./Care_Duration_Prediction_Model/Modeling/{file_name}/Final"
os.makedirs(final_save_dir, exist_ok=True)
final_path = os.path.join(final_save_dir, "bottleneck_output.csv")
result_df.to_csv(final_path, index=False, encoding='utf-8-sig')

################################################################################################
####################################### Final Result Analysis ##################################
################################################################################################

labels = ['성별_요양일', '수술여부_요양일', '연령대_요양일', '지역본부_요양일']

# 1. 산점도 시각화 (각 요양일별)
for label in labels:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=label, y='bottleneck_output', data=result_df, alpha=0.5)
    plt.title(f"📈 {label} vs Bottleneck 출력")
    plt.xlabel(label)
    plt.ylabel("Bottleneck 출력")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{final_save_dir}/scatter_{label}_vs_bottleneck.png")
    plt.close()

# 2. 히스토그램 (각 요양일 + bottleneck 분포 비교)
plt.figure(figsize=(12, 10))
for i, label in enumerate(labels):
    plt.subplot(3, 2, i + 1)
    sns.histplot(result_df[label], kde=True, color="skyblue")
    plt.title(f"📊 {label} 분포")

# bottleneck 출력 분포
plt.subplot(3, 2, 5)
sns.histplot(result_df['bottleneck_output'], kde=True, color="salmon")
plt.title("📊 Bottleneck 출력 분포")

plt.tight_layout()
plt.savefig(f"{final_save_dir}/histograms_각요양일_bottleneck.png")
plt.close()

# 3. 상관계수 분석
print("📊 상관계수 분석 결과 (Bottleneck vs 각 요양일)")
print("=" * 50)
for label in labels:
    pearson_corr, _ = pearsonr(result_df[label], result_df['bottleneck_output'])
    spearman_corr, _ = spearmanr(result_df[label], result_df['bottleneck_output'])
    print(f"🔹 {label}")
    print(f"  - Pearson  : {pearson_corr:.4f}")
    print(f"  - Spearman : {spearman_corr:.4f}")
    print("-" * 50)


################################################################################################
####################################### Final Result Rescaled ##################################
################################################################################################

loader = DataLoader(full_dataset, batch_size=len(full_dataset))
batch = next(iter(loader))
targets = batch[2]
print(targets.shape)

weights_mean = model.network[22].weight.data.mean().double().item()
biases_mean = model.network[22].bias.data.mean().double().item()
weights_mean = np.float64(weights_mean)
biases_mean = np.float64(biases_mean)
print("weights_mean =", weights_mean)
print("biases_mean =", biases_mean)
print("mean:", scaler.mean_)
print("std:", scaler.scale_)
mean_avg = scaler.mean_.mean()
std_avg = scaler.scale_.mean()


bottleneck_adjust = bottleneck_all.flatten() * weights_mean + biases_mean
bottleneck_adjust_inverse = bottleneck_adjust * std_avg + mean_avg



result_df = pd.DataFrame({
    "병명": full_diseases,
    "성별": full_sex,
    "연령대": full_age,
    "지역본부": full_region,
    "성별_요양일": targets[:, 0],
    "수술여부_요양일": targets[:, 1],
    "연령대_요양일": targets[:, 2],
    "지역본부_요양일": targets[:, 3],
    "bottleneck_output": bottleneck_all.flatten(),
    "성별_요양일_inverse": full_y_original[:, 0],
    "수술여부_요양일_inverse": full_y_original[:, 1],
    "연령대_요양일_inverse": full_y_original[:, 2],
    "지역본부_요양일_inverse": full_y_original[:, 3],
    "bottleneck_output_inverse": bottleneck_adjust_inverse
})
# 저장
final_save_dir = f"./Care_Duration_Prediction_Model/Modeling/{file_name}/Final"
os.makedirs(final_save_dir, exist_ok=True)
final_path = os.path.join(final_save_dir, "bottleneck_output_2.csv")
result_df.to_csv(final_path, index=False, encoding='utf-8-sig')


labels = ['성별_요양일', '수술여부_요양일', '연령대_요양일', '지역본부_요양일']

# 1. 산점도 시각화 (각 요양일별)
for i, label in enumerate(labels):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=full_y_original[:, i], y=bottleneck_adjust_inverse, alpha=0.5)
    plt.title(f"📈 {label} vs Rescaled_Bottleneck 출력")
    plt.xlabel(label)
    plt.ylabel("Bottleneck 출력")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{final_save_dir}/scatter_{label}_vs_rescaled_bottleneck.png")
    plt.close()

# 2. 히스토그램 (각 요양일 + bottleneck 분포 비교)
plt.figure(figsize=(12, 10))
for i, label in enumerate(labels):
    plt.subplot(3, 2, i + 1)
    sns.histplot(full_y_original[:, i], kde=True, color="skyblue")
    plt.title(f"📊 {label} 분포")

# bottleneck 출력 분포
plt.subplot(3, 2, 5)
sns.histplot(bottleneck_adjust_inverse, kde=True, color="salmon")
plt.title("📊 Rescaled_Bottleneck 출력 분포")

plt.tight_layout()
plt.savefig(f"{final_save_dir}/histograms_각요양일_Rescaled_bottleneck.png")
plt.close()

# 3. 상관계수 분석
print("📊 상관계수 분석 결과 (Rescaled_Bottleneck vs 각 요양일)")
print("=" * 50)
for i, label in enumerate(labels):
    pearson_corr, _ = pearsonr(full_y_original[:, i], bottleneck_adjust_inverse)
    spearman_corr, _ = spearmanr(full_y_original[:, i], bottleneck_adjust_inverse)
    print(f"🔹 {label}")
    print(f"  - Pearson  : {pearson_corr:.4f}")
    print(f"  - Spearman : {spearman_corr:.4f}")
    print("-" * 50)