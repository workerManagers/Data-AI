import os
from google.cloud import vision
import io

# 🔑 JSON 키 환경변수 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/hwangtaeeon/Downloads/ocr-project-455602-e1c8edaf1afb.json"

# 📂 이미지 폴더 경로
image_folder = "/Users/hwangtaeeon/Documents/workerManagers/산재요양 실태 보고서 (전처리)/jpg"

# 📁 OCR 결과 저장 폴더 (지정된 위치)
output_folder = "/Users/hwangtaeeon/Documents/workerManagers/산재요양 실태 보고서 (전처리)/google ocr"
os.makedirs(output_folder, exist_ok=True)

# Google Vision 클라이언트 생성
client = vision.ImageAnnotatorClient()

# 이미지 확장자 목록
image_extensions = ('.jpg', '.jpeg', '.png')

# 폴더 내 이미지 처리 반복
for filename in os.listdir(image_folder):
    if filename.lower().endswith(image_extensions):
        image_path = os.path.join(image_folder, filename)

        # 이미지 파일 읽기
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        # OCR 요청
        response = client.text_detection(image=image)
        texts = response.text_annotations

        # 결과 추출
        if texts:
            full_text = texts[0].description
            print(f"\n📄 {filename} OCR 결과:\n{full_text}")
        else:
            full_text = "[텍스트 없음]"
            print(f"\n📄 {filename} - 텍스트 없음")

        # 결과 저장
        txt_filename = os.path.splitext(filename)[0] + "_ocr.txt"
        output_path = os.path.join(output_folder, txt_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        print(f"📝 저장 완료: {output_path}")
