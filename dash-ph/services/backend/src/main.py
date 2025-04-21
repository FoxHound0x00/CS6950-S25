from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Response
import collections
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return "N!"

@app.get("/config")
def config():
    with open("src/ph_data_all_syn.json", "r") as f:
        data = json.load(f)

    with open("src/config.json", "r") as file:
        config_data = json.load(file)
        # config_data["Distance Metric"] = [keys for keys in data.keys()]
        config_data["Distance Metric"] = list(data.keys())
        return config_data
            
@app.get("/data/{metric}")
async def get_data(metric: str):
    with open("src/ph_data_all_syn.json", "r") as f:
        data = json.load(f)
    if metric in data:
        return data[metric]
    return {}
    
    
@app.get("/cluster_data/{metric}")
async def get_cluster_data(metric: str):
    with open("src/ph_data_all_syn.json", "r") as f:
        data = json.load(f)
        cluster_data = {}
        for key in data[metric].keys():
            cluster_data[key] = collections.Counter(data[metric][key])
            # cluster_data[key] = len(set(data[metric][key]))
        return cluster_data

@app.get("/pca/{metric}")
async def get_pca(metric: str):
 return {"message": "PCA data not available."}