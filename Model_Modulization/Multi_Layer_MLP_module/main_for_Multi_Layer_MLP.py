import sys
import os
import pandas as pd
pd.set_option('display.max_columns', None)
sys.path.append(os.path.abspath("./Model_Modulization/Multi_Layer_MLP_module/"))
from Multi_Layer_MLP_module import initialize, pipeline
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

act = 'softplus'
initialize(act)


input = pd.read_csv(f'./Care_Duration_Prediction_Model/Modeling/MLP_{act}/Final/bottleneck_output_2.csv')[['병명', '성별','수술여부', '연령대', '지역본부','bottleneck_output_inverse']]
final = []

for _, row in input.iterrows():
    disease = row['병명']
    sex = row['성별']
    surgery = row['수술여부']
    age = row['연령대']
    region = row['지역본부']
    bottleneck_output_inverse = row['bottleneck_output_inverse']

    result = pipeline(disease, sex, surgery, age, region)
    final.append((float(result[0]), bottleneck_output_inverse))

# 리스트 -> DataFrame 변환
final_df = pd.DataFrame(final, columns=['예측값', '정답값'])

# 저장 경로 설정
save_path = f'./Model_Modulization/Multi_Layer_MLP_module/MLP_{act}_prediction_result.csv'

# CSV로 저장
final_df.to_csv(save_path, index=False, encoding='utf-8-sig')

print(f"✅ 결과 저장 완료: {save_path}")
