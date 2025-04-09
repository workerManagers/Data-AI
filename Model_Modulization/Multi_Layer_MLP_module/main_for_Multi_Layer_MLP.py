import sys
import os
sys.path.append(os.path.abspath("./Model_Modulization/Multi_Layer_MLP_module/"))
from Multi_Layer_MLP_module import initialize, pipeline

def main():
    # 1. 초기화
    initialize()

    # 2. 예측할 입력값 지정
    disease = "S000(머리덮개의 얕은 손상)"
    sex = "여자"
    surgery = "예"
    age = "30-39세"
    region = "서울지역"

    # 3. 파이프라인 실행
    result = pipeline(disease, sex, surgery, age, region)

    # 4. 출력
    print("📌 최종 예측된 요양일 (Rescaled Bottleneck Output):", float(result[0]))

if __name__ == "__main__":
    main()
