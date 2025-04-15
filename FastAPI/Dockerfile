FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# ↓↓↓ 이 줄을 추가하세요!
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('jhgan/ko-sbert-sts')"

COPY . /code

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
