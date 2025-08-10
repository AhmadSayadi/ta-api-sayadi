# app.py — FastAPI: Prediksi murni model ML dengan API-Key header (hardcoded)
import os
import joblib
import pandas as pd
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# ====== Konfigurasi ======
API_KEY = "a3c9f9c09c8e4b78a5fdf402d77b92de" 

MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "svm_haz_pmk.pkl")
ENC_PATH   = os.path.join(MODEL_DIR, "label_encoder.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENC_PATH):
    raise RuntimeError(
        "Model/encoder tidak ditemukan. Pastikan file ada di folder 'models/'.\n"
        f"MODEL_PATH: {MODEL_PATH}\nENC_PATH  : {ENC_PATH}"
    )

# ====== Load model & encoder ======
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENC_PATH)
CLASSES = list(encoder.classes_)

# ====== FastAPI init + CORS ======
app = FastAPI(
    title="API Prediksi TB/U (ML-only)",
    description="Endpoint /predict & /predict-batch: murni prediksi model (tanpa WHO/PMK rules).",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ====== Auth via Header (hardcoded key) ======
def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key is None or x_api_key != API_KEY:
        # 401 Unauthorized
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# ====== Schemas ======
class Sample(BaseModel):
    usia_bulan: int
    jk_bin: Optional[int] = None        # 0=laki-laki, 1=perempuan
    jenis_kelamin: Optional[str] = None # alternatif input jk
    tinggi_cm: float
    berat_kg: float

    @field_validator("jk_bin")
    @classmethod
    def _valid_jk_bin(cls, v):
        if v is not None and v not in (0, 1):
            raise ValueError("jk_bin harus 0 (laki-laki) atau 1 (perempuan)")
        return v

    @field_validator("jenis_kelamin")
    @classmethod
    def _normalize_gender(cls, v):
        if v is None:
            return v
        s = str(v).strip().lower()
        mapping = {
            "laki-laki": "laki-laki",
            "laki laki": "laki-laki",
            "laki2": "laki-laki",
            "laki": "laki-laki",
            "perempuan": "perempuan",
            "wanita": "perempuan",
            "pr": "perempuan",
        }
        if s not in mapping:
            raise ValueError("jenis_kelamin harus 'Laki-laki' atau 'Perempuan'")
        return mapping[s]

    def to_feature_row(self) -> Dict[str, Any]:
        if self.jk_bin is not None:
            jk_bin_val = int(self.jk_bin)
        elif self.jenis_kelamin is not None:
            jk_bin_val = 0 if self.jenis_kelamin == "laki-laki" else 1
        else:
            raise ValueError("Sertakan jk_bin ATAU jenis_kelamin.")
        return {
            "usia_bulan": int(self.usia_bulan),
            "jk_bin": jk_bin_val,
            "tinggi_cm": float(self.tinggi_cm),
            "berat_kg": float(self.berat_kg),
        }

# ====== Utils ======
def _predict_df(df: pd.DataFrame):
    y_idx = model.predict(df)
    labels = encoder.inverse_transform(y_idx)
    probs = None
    if hasattr(model, "predict_proba"):
        try:
            P = model.predict_proba(df)
            probs = [{CLASSES[i]: float(P[r, i]) for i in range(len(CLASSES))}
                     for r in range(P.shape[0])]
        except Exception:
            probs = None
    return y_idx, labels, probs

# ====== Endpoints ======
@app.get("/health")
def health():
    return {"status": "ok", "classes": CLASSES}

@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict_only(item: Sample):
    try:
        row = item.to_feature_row()
        X = pd.DataFrame([row])
        y_idx, labels, probs = _predict_df(X)
        return {
            "success": True,
            "message": "Prediction success",
            "data": {
                "usia_bulan": row["usia_bulan"],
                "jk_bin": row["jk_bin"],
                "tinggi_cm": row["tinggi_cm"],
                "berat_kg": row["berat_kg"],
                "status": labels[0],
                "probabilities": probs[0] if probs else None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict-batch", dependencies=[Depends(verify_api_key)])
def predict_batch(items: List[Sample]):
    try:
        rows = [it.to_feature_row() for it in items]
        X = pd.DataFrame(rows)
        y_idx, labels, probs = _predict_df(X)
        results = []
        for i in range(len(labels)):
            results.append({
                **rows[i],
                "status": labels[i],
                "probabilities": probs[i] if probs else None
            })
        return {
            "success": True,
            "message": "Batch prediction success",
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
