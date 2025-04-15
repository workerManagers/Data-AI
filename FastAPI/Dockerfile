FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

ENV TRANSFORMERS_CACHE=/code/model_cache
ENV HF_HOME=/code/model_cache

# 미리 모델 다운로드 (main2.py에서만 필요하지만, main.py에서 영향 없음)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('jhgan/ko-sbert-sts')"

COPY . /code

# 기본값은 main.py로, Cloudtype에서 필요시 override
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
