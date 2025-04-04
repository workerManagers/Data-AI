import pytesseract
from PIL import Image
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image_folder = './Data preprocessing/raw_img_data'
output_folder = './Data preprocessing/OCR_result/tesseract_results'
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        image_path = os.path.join(image_folder, filename)
        text = pytesseract.image_to_string(Image.open(image_path), lang='kor')
        with open(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_tesseract.txt"), 'w', encoding='utf-8') as f:
            f.write(text)