FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 캐시 경로 고정
ENV TRANSFORMERS_CACHE=/code/model_cache
ENV HF_HOME=/code/model_cache

# 미리 모델 다운로드
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('jhgan/ko-sbert-sts')"

COPY . /code

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
