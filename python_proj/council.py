from fastapi import FastAPI
import requests
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allow all origins
    allow_credentials=False,      # MUST be False with "*"
    allow_methods=["*"],          # Allow all HTTP methods
    allow_headers=["*"],          # Allow all headers
    expose_headers=["*"],         # Expose all headers to the client
)

class laptop(BaseModel):
    query:str
    model_name:str
    stream:bool

@app.post("/tsdc/models")
def laptop(payload:laptop):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": payload.model_name,
            "prompt": payload.query,
            "stream": payload.stream
        },
        timeout=300
    )
    response.raise_for_status()

    data = response.json()
    return data




class vision(BaseModel):
    query:str

@app.post('/tsdc/vision')
def status(payload:vision):
    return {"response":f"Sab changa si!!! Received {payload.query}"}