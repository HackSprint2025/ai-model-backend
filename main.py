from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import PredictionRequest, PredictionResponse
from app.predict import load_model, predict

app = FastAPI(title="24-feature model API")

FRONTEND_ORIGINS = ["*"]  

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    try:
        load_model()
    except Exception as e:

        raise RuntimeError(f"Failed to load model on startup: {e}")

@app.get("/")
def root():
    return {"message": "24-feature model API. POST /predict with 24 floats."}

@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(req: PredictionRequest):
    try:
        pred, conf = predict(req.features)
        return {"prediction": pred, "confidence": conf}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
