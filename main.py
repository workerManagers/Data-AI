from typing import Union
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel
import os
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import openai
from openai import OpenAI
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import emergency as em
from datetime import datetime
import sqlite3
import warnings
from geopy.geocoders import Nominatim
warnings.filterwarnings("ignore", category=DeprecationWarning)

geo_local = Nominatim(user_agent='South Korea')

def geocoding(address):
    try:
        geo = geo_local.geocode(address)
        x, y = geo.latitude, geo.longitude
        return x, y

    except:
        return 0, 0

path = './'
emergency = pd.read_csv(path + '응급실 정보.csv')

app = FastAPI()

def insert_db(addr, text, lat, lon, em_class, em_df):

    path = './db/em.db'
    conn = sqlite3.connect(path)

    dt = datetime.now()
    dt = dt.strftime('%Y-%m-%d %H:%M:%S')

    hospitals = [''] * 3
    addresses = [''] * 3
    tels = [''] * 3

    for i in range(min(len(em_df), 3)):
        hospitals[i] = em_df.iloc[i].get('병원이름', '')
        addresses[i] = em_df.iloc[i].get('주소', '')
        tels[i] = em_df.iloc[i].get('전화번호 1', '')

    data = pd.DataFrame({
        'datetime': dt,
        'input_text': text,
        'input_addr': addr,
        'input_latitude': lat,
        'input_longitude': lon,
        'em_class': em_class,
        'hospital1': hospitals[0],
        'addr1': addresses[0],
        'tel1': tels[0],
        'hospital2': hospitals[1],
        'addr2': addresses[1],
        'tel2': tels[1],
        'hospital3': hospitals[2],
        'addr3': addresses[2],
        'tel3': tels[2]
    }, index=[0])

    data.to_sql('em', conn, if_exists='append', index=False)
    conn.close()
    
def insert_db_norm(addr, text, lat, lon, em_class,first_aid):

    path = './db/norm.db'
    conn = sqlite3.connect(path)

    dt = datetime.now()
    dt = dt.strftime('%Y-%m-%d %H:%M:%S')



    data = pd.DataFrame({
        'datetime': dt,
        'input_text': text,
        'input_addr': addr,
        'input_latitude': lat,
        'input_longitude': lon,
        'em_class': em_class,
        'first_aid': first_aid
    }, index=[0])

    data.to_sql('norm', conn, if_exists='append', index=False)
    conn.close()


class HospitalRecommendation(BaseModel):
    hospital_name: str
    address: str
    contact: str
    road_distance_km: float
    time: int
    hospital_lat: float
    hospital_lng: float
    hospital_class: list[str]
    image: Union[str,None] = None

class HospitalResponse(BaseModel):
    text: str
    emergency_class: int
    first_aid: str
    input_lat : float
    input_lng : float
    recommendations: list[HospitalRecommendation]
    
    
    

@app.post("/hospital_by_module", response_model=HospitalResponse)
async def get_hospital_by_module(
    file: UploadFile = File(...),
    # latitude: float = Query(..., description="위도"),
    # longitude: float = Query(..., description="경도"),
    address: str = Query(..., description="도로명 주소"),
    emlc_value: int = Query(3, descrpition="응급실 개수")
):
    try:
        latitude, longitude = geocoding(address)
        openai.api_key = em.load_file(path + 'api_key.txt')
        os.environ['OPENAI_API_KEY'] = openai.api_key
        map_key = em.load_file(path + 'map_key.txt')
        map_key = json.loads(map_key)
        c_id, c_key = map_key['c_id'], map_key['c_key']

        save_directory = path + "fine_tuned_bert"
        model = AutoModelForSequenceClassification.from_pretrained(save_directory)
        tokenizer = AutoTokenizer.from_pretrained(save_directory)


        result = em.audio2text(file)

        summary_result,_ = em.text_summary(result,split=True)

        predicted_class, _ = em.predict(summary_result, model, tokenizer)

        first_aid = ''
        
        if predicted_class <= 2:
            result = em.recommendation(latitude, longitude, emergency, c_id, c_key,emlc_value)
            first_aid = em.recommend_firstaid(summary_result)
            insert_db(address, summary_result,latitude,longitude,predicted_class+1,result)
        else:
            first_aid = em.recommend_firstaid(summary_result)
            insert_db_norm(address, summary_result,latitude,longitude,predicted_class+1,first_aid)
            
        recommendations = []
        if isinstance(result, pd.DataFrame):
            for _, row in result.iterrows():
                recommendations.append(
                    HospitalRecommendation(
                        hospital_name=row.get("병원이름", ""),
                        address=row.get("주소", ""),
                        contact=row.get("전화번호 1", ""),
                        road_distance_km=row.get("도로 거리 (km)", 0.0),
                        time=int(row.get("이동 시간 (분)",0.0)),
                        hospital_lat=row.get("위도",0.0),
                        hospital_lng=row.get("경도",0.0),
                        hospital_class=row.get("진료과목", "").split(","),
                        image=row.get("병원이름", "")
                    )
                )

        return HospitalResponse(text=summary_result,emergency_class=predicted_class+1,first_aid=first_aid,recommendations=recommendations,input_lat=latitude,input_lng=longitude)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    



@app.get("/emergencydata")
def show_data():
    path = './db/em.db'

    conn = sqlite3.connect(path)

    try:
        df = pd.read_sql('SELECT * FROM em', conn)
        df = df.sort_values(by='datetime', ascending=False)
        data_json = df.to_json(orient="records", force_ascii=False)

    finally:
        conn.close()

    return {
        "emergency_data": data_json
    }
    
@app.get("/normdata")
def show_data1():
    path = './db/norm.db'
    
    conn = sqlite3.connect(path)
    
    try:
        df = pd.read_sql('SELECT * FROM norm', conn)
        df = df.sort_values(by='datetime', ascending=False)
        data_json = df.to_json(orient="records", force_ascii=False)
        
    finally:
        conn.close()
    
    return {
        "norm_data": data_json
    }