import os
import re
import pandas as pd

# 컬럼 정의
columns = [
    "병명", "성별_남자", "성별_여자", "수술여부_아니오", "수술여부_예",
    "연령대_30세미만", "연령대_30-39세", "연령대_40-49세", "연령대_50-59세", "연령대_60세이상",
    "지역본부_서울지역", "지역본부_부산지역", "지역본부_대구지역", "지역본부_경인지역", "지역본부_광주지역", "지역본부_대전지역"
]

# 텍스트 파일들이 있는 폴더
folder_path = "./Data preprocessing/OCR_result/google_ocr_results"

# 결과 저장용 리스트
data = []

# 각 파일 처리
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
            text = f.read()

        # 한 행 초기화
        row = dict.fromkeys(columns, None)

        # 병명 추출 (ex: 1. C510(벨마비(Bell's palsy)))
        disease_match = re.search(r"\d+\.\s*(\S+)\(", text)
        if disease_match:
            row["병명"] = disease_match.group(1).strip()

        # 성별 평균
        male_avg = re.search(r"남자\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+([\d.]+)", text)
        female_avg = re.search(r"여자\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+([\d.]+)", text)
        row["성별_남자"] = float(male_avg.group(1)) if male_avg else None
        row["성별_여자"] = float(female_avg.group(1)) if female_avg else None

        # 수술여부 평균 (있으면 추출, 없으면 None)
        op_no_avg = re.search(r"수술[^\n]*아니오\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+([\d.]+)", text)
        op_yes_avg = re.search(r"수술[^\n]*예\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+([\d.]+)", text)
        row["수술여부_아니오"] = float(op_no_avg.group(1)) if op_no_avg else None
        row["수술여부_예"] = float(op_yes_avg.group(1)) if op_yes_avg else None

        # 연령대 평균
        age_groups = {
            "30세미만": "연령대_30세미만",
            "30-39세": "연령대_30-39세",
            "40-49세": "연령대_40-49세",
            "50-59세": "연령대_50-59세",
            "60세이상": "연령대_60세이상"
        }
        for label, col_name in age_groups.items():
            match = re.search(rf"{label}\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+([\d.]+)", text)
            if match:
                row[col_name] = float(match.group(1))

        # 지역본부 평균
        regions = {
            "서울지역": "지역본부_서울지역",
            "부산지역": "지역본부_부산지역",
            "대구지역": "지역본부_대구지역",
            "경인지역": "지역본부_경인지역",
            "광주지역": "지역본부_광주지역",
            "대전지역": "지역본부_대전지역"
        }
        for label, col_name in regions.items():
            match = re.search(rf"{label}\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+([\d.]+)", text)
            if match:
                row[col_name] = float(match.group(1))

        # 한 행 추가
        data.append(row)

# 데이터프레임 생성
df = pd.DataFrame(data, columns=columns)

# CSV로 저장
df.to_csv('./Data preprocessing/Regular_Expression/Regular_Expression_result/상병별_요양기간_평균값.csv', index=False, encoding="utf-8-sig")
