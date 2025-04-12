import pandas as pd
import torch.nn as nn
import joblib
import json
import numpy as np
import torch

def initialize(root_path):
    global model, le, scaler, expected_columns
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
            layers.append(nn.Softplus())
            layers.append(nn.Linear(bottleneck_dim, output_dim))
            self.network = nn.Sequential(*layers)
        def forward(self, disease_ids, features):
            embedded = self.embedding(disease_ids)  # (batch, embedding_dim)
            x = torch.cat((embedded, features), dim=1)
            return self.network(x)
    num_diseases = 171
    embedding_dim = 16
    input_dim = 15
    hidden_dims = [256, 128, 128, 64, 64]
    bottleneck_dim = 1
    output_dim = 4
    model = ComplexMLPWithEmbedding(num_diseases, embedding_dim, input_dim, hidden_dims, bottleneck_dim, output_dim)
    model.load_state_dict(
        torch.load(
            f'{root_path}/Multi_Layer_MLP_softplus.pt',
            map_location=torch.device('cuda'),
            weights_only=True
        )
    )
    le = joblib.load(f'{root_path}/Multi_Layer_MLP_softplus_LabelEncoder.pkl')
    scaler = joblib.load(f'{root_path}/Multi_Layer_MLP_softplus_StandardScaler.pkl')
    with open(f'{root_path}/Multi_Layer_MLP_softplus_input_columns.json', 'r', encoding='utf-8') as f:
        expected_columns = json.load(f)

def Data_preprocessing(disease,sex,surgery,age,region):
    df = pd.DataFrame([{
        '병명': disease,
        '성별': sex,
        '수술여부': surgery,
        '연령대': age,
        '지역본부': region
    }])
    if disease not in le.classes_:
        raise ValueError(f"'{disease}'는 등록되지 않은 병명입니다.")
    df['병명'] = le.transform([disease])
    df = pd.get_dummies(df, columns=['성별', '수술여부', '연령대', '지역본부'])
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    df = df.reindex(columns=expected_columns, fill_value=0)
    df = df.replace({True: 1, False: 0}).astype(int)
    disease_tensor = torch.tensor(df['병명'].values, dtype=torch.long)
    feature_tensor = torch.tensor(df.drop(columns=['병명']).values, dtype=torch.float32)
    return disease_tensor, feature_tensor

def run_model_with_bottleneck(disease_tensor, feature_tensor, model):
    model.eval()
    bottleneck_outputs = []
    hook_layer = model.network[21]

    def save_bottleneck_output(module, input, output):
        bottleneck_outputs.append(output.detach().cpu().numpy())

    hook_handle = hook_layer.register_forward_hook(save_bottleneck_output)

    with torch.no_grad():
        _ = model(disease_tensor, feature_tensor).cpu().numpy()

    hook_handle.remove()
    bottleneck = np.vstack(bottleneck_outputs).flatten()
    return  bottleneck

def softplus_adjust_bottleneck_output(model,scaler,bottleneck):
    weight_mean = np.float64(model.network[-1].weight.data.cpu().numpy().mean())
    bias_mean = np.float64(model.network[-1].bias.data.cpu().numpy().mean())
    mean_avg = np.float64(scaler.mean_.mean())
    std_avg = np.float64(scaler.scale_.mean())

    bottleneck_adjust = bottleneck * weight_mean + bias_mean
    bottleneck_inverse = bottleneck_adjust * std_avg + mean_avg

    return bottleneck_inverse

def pipeline(disease,sex,surgery,age,region):
    disease_tensor, feature_tensor = Data_preprocessing(disease, sex, surgery, age, region)
    bottleneck = run_model_with_bottleneck(disease_tensor, feature_tensor, model)
    bottleneck_inverse = softplus_adjust_bottleneck_output(model, scaler, bottleneck)
    return bottleneck_inverse