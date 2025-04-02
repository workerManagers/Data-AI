import os
import pytesseract
from PIL import Image, ImageEnhance
from natsort import natsorted

# 경로 설정
image_folder = "/Users/hwangtaeeon/Downloads/데이터마이닝 기법을 활용한 상병별 산재요양 실태 분석-부록3"
output_folder = "/Users/hwangtaeeon/Downloads/산재요양 실태 보고서 v5"
os.makedirs(output_folder, exist_ok=True)

# 자연 정렬된 이미지 파일 목록
image_files = natsorted([
    f for f in os.listdir(image_folder)
    if f.lower().endswith(('.jpg', '.png'))
])

# 키워드 목록 (OCR 후 텍스트 안에 포함되는지 확인)
keywords = [
    "기초", "기추", "동계량", "통계량", "기초통계량", "기초 통계",
    "통계량", "통게량", '기초통게량', '서울지역', '부산지역', '여자', '남자', '기초통계랑'
]

# OCR 수행 및 키워드 감지
for filename in image_files:
    path = os.path.join(image_folder, filename)
    
    # 이미지 열고 전처리
    image = Image.open(path).convert('L')  # Grayscale
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)  # 대비 증가

    # OCR 수행
    text = pytesseract.image_to_string(image, lang='kor+eng').lower()

    # 결과 출력 (선택적으로 유지)
    print(f"\n--- OCR 결과: {filename} ---")
    print(text)

    # 키워드 포함 여부 판단
    if any(kw in text for kw in keywords):
        print(f"✔ 저장: {filename}")
        # 키워드가 있을 경우 이미지만 저장
        image.save(os.path.join(output_folder, filename))
    else:
        print(f"✘ 건너뜀: {filename}")
