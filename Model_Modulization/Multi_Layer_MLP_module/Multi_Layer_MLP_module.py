import pandas as pd
import torch.nn as nn
import joblib
import json
import numpy as np
import torch

def initialize():
    print('model loading...')
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
    model.load_state_dict(torch.load('./Model_Modulization/Multi_Layer_MLP_module/Multi_Layer_MLP.pt', map_location=torch.device('cuda')))
    print('LabelEncoder loading...')
    le = joblib.load('./Model_Modulization/Multi_Layer_MLP_module/Multi_Layer_MLP_LabelEncoder.pkl')
    print('StandardScaler loading...')
    scaler = joblib.load('./Model_Modulization/Multi_Layer_MLP_module/Multi_Layer_MLP_StandardScaler.pkl')
    print('columns for get_dummies loading...')
    with open('./Model_Modulization/Multi_Layer_MLP_module/Multi_Layer_MLP_input_columns.json', 'r', encoding='utf-8') as f:
        expected_columns = json.load(f)
    print('load Complete!')


def Data_preprocessing(disease,sex,surgery,age,region):
    df = pd.DataFrame([{
        '병명': disease,
        '성별': sex,
        '수술여부': surgery,
        '연령대': age,
        '지역본부': region
    }])
    print('Data preprocessing start...')
    df['병명'] = le.transform(df['병명'])
    df = pd.get_dummies(df, columns=['성별', '수술여부', '연령대', '지역본부'])
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_columns]

    disease_tensor = torch.tensor(df['병명'].values, dtype=torch.long)
    feature_tensor = torch.tensor(df.drop(columns=['병명']).values, dtype=torch.float32)
    print('Data preprocessing complete!')
    return disease_tensor, feature_tensor

def run_model_with_bottleneck(disease_tensor, feature_tensor, model):
    print('model running...')
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
    print('model run complete!')
    return  bottleneck

def adjust_bottleneck_output(model,scaler,bottleneck):
    print('bottleneck adjusting...')
    final_layer = model.network[-1]  # Linear layer
    weight_mean = final_layer.weight.data.mean().item()
    bias_mean = final_layer.bias.data.mean().item()
    mean_avg = scaler.mean_.mean()
    std_avg = scaler.scale_.mean()

    bottleneck_adjust = bottleneck * weight_mean + bias_mean
    bottleneck_inverse = bottleneck_adjust * std_avg + mean_avg

    print('bottleneck adjust complete1')
    return bottleneck_inverse

def pipeline(disease,sex,surgery,age,region):
    disease_tensor, feature_tensor = Data_preprocessing(disease, sex, surgery, age, region)
    bottleneck = run_model_with_bottleneck(disease_tensor, feature_tensor, model)
    bottleneck_inverse = adjust_bottleneck_output(model, scaler, bottleneck)
    return bottleneck_inverse