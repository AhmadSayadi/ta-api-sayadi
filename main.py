# app.py — FastAPI: Prediksi murni model ML + endpoint test private/public
import os
import joblib
import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# ====== Konfigurasi ======
API_KEY = "a3c9f9c09c8e4b78a5fdf402d77b92de"

MODEL_DIR = os.getenv("MODEL_DIR", "model")
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

    @field_validator("usia_bulan")
    @classmethod
    def _validate_usia_bulan(cls, v):
        if v is None or v == 0:
            raise ValueError("usia_bulan tidak boleh kosong atau 0")
        if v < 0:
            raise ValueError("usia_bulan harus bernilai positif")
        return v

    @field_validator("tinggi_cm")
    @classmethod
    def _validate_tinggi_cm(cls, v):
        if v is None or v == 0:
            raise ValueError("tinggi_cm tidak boleh kosong atau 0")
        if v < 0:
            raise ValueError("tinggi_cm harus bernilai positif")
        return v

    @field_validator("berat_kg")
    @classmethod
    def _validate_berat_kg(cls, v):
        if v is None or v == 0:
            raise ValueError("berat_kg tidak boleh kosong atau 0")
        if v < 0:
            raise ValueError("berat_kg harus bernilai positif")
        return v

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
        if not s:  # Validasi string kosong
            raise ValueError("jenis_kelamin tidak boleh kosong")
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

def _status_payload(desc: str) -> Dict[str, Any]:
    return {
        "status": "ok",
        "message": "Server is running",
        "time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "desc": desc,  # "Private" atau "Public"
    }

# ====== Endpoints ======
@app.get("/health")
def health():
    return {"status": "ok", "classes": CLASSES}

# --- Test status (tanpa API key) ---
@app.get("/public")
def public_status():
    return _status_payload("Public")

# --- Test status (dengan API key) ---
@app.get("/private", dependencies=[Depends(verify_api_key)])
def private_status():
    return _status_payload("Private")

@app.get("/model")
def model_info():
    """Endpoint untuk menampilkan informasi model ML"""
    try:
        # Ambil informasi dari model yang sudah dimuat
        model_type = type(model).__name__
        
        # Cek apakah model memiliki atribut tertentu
        model_params = {}
        if hasattr(model, 'get_params'):
            model_params = model.get_params()
        
        # Ambil informasi kernel jika SVM
        kernel_info = "Unknown"
        if hasattr(model, 'kernel'):
            kernel_info = model.kernel
        
        # Ambil feature names jika ada
        feature_names = ["usia_bulan", "jk_bin", "tinggi_cm", "berat_kg"]
        if hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)
        
        # Ambil jumlah support vectors jika SVM
        n_support = None
        if hasattr(model, 'n_support_'):
            n_support = model.n_support_.tolist() if hasattr(model.n_support_, 'tolist') else list(model.n_support_)
        
        return {
            "model_name": "SVM HAZ PMK",
            "model_type": model_type,
            "classes": CLASSES,
            "n_classes": len(CLASSES),
            "model_parameters": {
                "kernel": kernel_info,
                "n_support_vectors": n_support,
                "total_support_vectors": sum(n_support) if n_support else None,
                "other_params": {k: str(v) for k, v in model_params.items() if k not in ['kernel']}
            },
            "model_features": feature_names,
            "model_info": {
                "algorithm": f"{model_type}",
                "encoder_classes": list(encoder.classes_) if hasattr(encoder, 'classes_') else None,
                "model_file": MODEL_PATH,
                "encoder_file": ENC_PATH
            },
            "note": "Untuk informasi performance metrics dan confusion matrix, lihat endpoint /performance-metrics"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.get("/performance-metrics")
async def performance_metrics():
    """Get model performance metrics"""
    try:
        return {
            "evaluation_metrics": {
                "precision": {
                    "class_0": 0.66,
                    "class_1": 0.79,
                    "macro_avg": 0.72,
                    "weighted_avg": 0.73
                },
                "recall": {
                    "class_0": 0.78,
                    "class_1": 0.67,
                    "macro_avg": 0.72,
                    "weighted_avg": 0.72
                },
                "f1_score": {
                    "class_0": 0.71,
                    "class_1": 0.72,
                    "macro_avg": 0.72,
                    "weighted_avg": 0.72
                },
                "support": {
                    "class_0": 27,
                    "class_1": 33,
                    "total": 60
                }
            },
            "confusion_matrix": {
                "matrix": [[21, 6], [11, 22]],
                "labels": ["class_0", "class_1"],
                "description": "Confusion Matrix (Actual vs Predicted)",
                "interpretation": {
                    "true_negatives": 21,
                    "false_positives": 6,
                    "false_negatives": 11,
                    "true_positives": 22
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting performance metrics: {str(e)}")

@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict_only(item: Sample):
    try:
        # Validasi tambahan untuk memastikan semua field terisi
        if item.usia_bulan is None or item.usia_bulan == 0:
            raise ValueError("usia_bulan tidak boleh kosong atau 0")
        if item.tinggi_cm is None or item.tinggi_cm == 0:
            raise ValueError("tinggi_cm tidak boleh kosong atau 0")
        if item.berat_kg is None or item.berat_kg == 0:
            raise ValueError("berat_kg tidak boleh kosong atau 0")
        if item.jk_bin is None and (item.jenis_kelamin is None or item.jenis_kelamin.strip() == ""):
            raise ValueError("jk_bin atau jenis_kelamin harus diisi")
            
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
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=f"Validation error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict-batch", dependencies=[Depends(verify_api_key)])
def predict_batch(items: List[Sample]):
    try:
        # Validasi setiap item dalam batch
        for idx, item in enumerate(items):
            if item.usia_bulan is None or item.usia_bulan == 0:
                raise ValueError(f"Item {idx+1}: usia_bulan tidak boleh kosong atau 0")
            if item.tinggi_cm is None or item.tinggi_cm == 0:
                raise ValueError(f"Item {idx+1}: tinggi_cm tidak boleh kosong atau 0")
            if item.berat_kg is None or item.berat_kg == 0:
                raise ValueError(f"Item {idx+1}: berat_kg tidak boleh kosong atau 0")
            if item.jk_bin is None and (item.jenis_kelamin is None or item.jenis_kelamin.strip() == ""):
                raise ValueError(f"Item {idx+1}: jk_bin atau jenis_kelamin harus diisi")
                
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
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=f"Validation error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
