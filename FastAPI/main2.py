import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
from typing import List, Dict, Any, Union

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

root_path = '.'
sys.path.append(os.path.abspath(f"{root_path}/"))

import Recommendation_based_on_similarity as similar

# 유사도 추천 모듈 초기화
logger.info("유사도 추천 모듈(Recommendation_based_on_similarity) 초기화 시작")
similar.initialize()
logger.info("유사도 추천 모듈(Recommendation_based_on_similarity) 초기화 완료")

app = FastAPI()

class SimilarityResult(BaseModel):
    id: Union[str, int]
    similarity: str

class CompareRequest(BaseModel):
    input_text: str
    dataset: List[Dict[str, Any]]

@app.post("/compare", response_model=List[SimilarityResult])
def compare_resume(data: CompareRequest):
    logger.info(f"/compare 호출: input_text 길이={len(data.input_text)}, dataset 크기={len(data.dataset)}")
    try:
        result = similar.pipeline(data.dataset, data.input_text)
        logger.info(f"/compare 결과: {result}")
        return result
    except Exception as e:
        logger.error(f"/compare Exception: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {str(e)}")

# 헬스체크 엔드포인트 추가 (Cloudtype용)
@app.get("/health")
def health_check():
    return {"status": "ok"}

# uvicorn main2:app --host=0.0.0.0 --port=8000
# http://127.0.0.1:8000/docs
