import easy_ocr
import os


reader = easyocr.Reader(['ko'])

image_folder = './Data preprocessing/raw_img_data'
output_folder = './Data preprocessing/OCR_result/easyocr_results'
os.makedirs(output_folder, exist_ok=True)

# 이미지 폴더 내의 모든 이미지에 대해 OCR 수행
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        image_path = os.path.join(image_folder, filename)
        result = reader.readtext(image_path)
        # 결과를 텍스트로 저장
        with open(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_easyocr.txt"), 'w', encoding='utf-8') as f:
            for _, text, _ in result:
                f.write(text + '\n')
