"""
api.py — Biological Firewall Engine: REST API
===============================================
FastAPI service exposing the fly-brain connectome classifier.

Endpoints:
  POST /api/v1/classify  — classify a packet from raw NSL-KDD features
  GET  /api/v1/health    — model health summary
  GET  /api/v1/metrics   — full training metrics

Run:
  uvicorn api:app --port 8503
"""

import json
import os
import pickle
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model import FlyBrainNet

# ═══════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]

KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty_level",
]

ATTACK_CATEGORIES = {
    "normal": 0, "back": 1, "land": 1, "neptune": 1, "pod": 1, "smurf": 1,
    "teardrop": 1, "mailbomb": 1, "apache2": 1, "processtable": 1, "udpstorm": 1,
    "ipsweep": 2, "nmap": 2, "portsweep": 2, "satan": 2, "mscan": 2, "saint": 2,
    "ftp_write": 3, "guess_passwd": 3, "imap": 3, "multihop": 3, "phf": 3,
    "spy": 3, "warezclient": 3, "warezmaster": 3, "xlock": 3, "xsnoop": 3,
    "snmpguess": 3, "snmpgetattack": 3, "httptunnel": 3, "sendmail": 3,
    "named": 3, "worm": 3,
    "buffer_overflow": 4, "loadmodule": 4, "perl": 4, "rootkit": 4,
    "xterm": 4, "ps": 4, "sqlattack": 4,
}


# ═══════════════════════════════════════════════════════════
# Pydantic Models
# ═══════════════════════════════════════════════════════════
class PacketFeatures(BaseModel):
    """NSL-KDD packet feature schema."""

    # Categorical features (will be one-hot encoded)
    protocol_type: str = Field(..., examples=["tcp"], description="Protocol type: tcp, udp, icmp")
    service: str = Field(..., examples=["http"], description="Network service on destination, e.g. http, ftp, smtp")
    flag: str = Field(..., examples=["SF"], description="Connection status flag, e.g. SF, S0, REJ")

    # 38 numeric features
    duration: float = 0
    src_bytes: float = 0
    dst_bytes: float = 0
    land: float = 0
    wrong_fragment: float = 0
    urgent: float = 0
    hot: float = 0
    num_failed_logins: float = 0
    logged_in: float = 0
    num_compromised: float = 0
    root_shell: float = 0
    su_attempted: float = 0
    num_root: float = 0
    num_file_creations: float = 0
    num_shells: float = 0
    num_access_files: float = 0
    num_outbound_cmds: float = 0
    is_host_login: float = 0
    is_guest_login: float = 0
    count: float = 1
    srv_count: float = 1
    serror_rate: float = 0.0
    srv_serror_rate: float = 0.0
    rerror_rate: float = 0.0
    srv_rerror_rate: float = 0.0
    same_srv_rate: float = 1.0
    diff_srv_rate: float = 0.0
    srv_diff_host_rate: float = 0.0
    dst_host_count: float = 1
    dst_host_srv_count: float = 1
    dst_host_same_srv_rate: float = 1.0
    dst_host_diff_srv_rate: float = 0.0
    dst_host_same_src_port_rate: float = 0.0
    dst_host_srv_diff_host_rate: float = 0.0
    dst_host_serror_rate: float = 0.0
    dst_host_srv_serror_rate: float = 0.0
    dst_host_rerror_rate: float = 0.0
    dst_host_srv_rerror_rate: float = 0.0


class ClassScore(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(..., description="Probability as a percentage (0–100)")


class ClassifyResponse(BaseModel):
    classification: str = Field(..., description="Predicted class name")
    class_id: int = Field(..., description="Predicted class index (0–4)")
    confidence: float = Field(..., description="Confidence of the predicted class (%)")
    verdict: str = Field(..., description="BENIGN or THREAT")
    scores: List[ClassScore] = Field(..., description="Confidence scores for all 5 classes")


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str = "FlyBrainNet"
    num_neurons: int
    num_synapses: int
    accuracy: float
    f1_macro: float


class MetricsResponse(BaseModel):
    """Full training metrics — dynamic keys, so we use a dict."""
    pass  # returned as raw dict


# ═══════════════════════════════════════════════════════════
# Engine State (populated on startup)
# ═══════════════════════════════════════════════════════════
class _EngineState:
    """Container for model artefacts loaded at startup."""

    model: Optional[FlyBrainNet] = None
    scaler: Optional[Any] = None
    train_columns: Optional[pd.Index] = None
    metrics: Optional[Dict[str, Any]] = None


_engine = _EngineState()


def _load_engine() -> None:
    """Load model, scaler, and training columns — mirrors app.py load_engine()."""

    # ── Training columns (derive from KDDTrain.txt) ──
    train_data_path = os.path.join(BASE_DIR, "KDDTrain.txt")
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Training data not found: {train_data_path}")

    df = pd.read_csv(train_data_path, names=KDD_COLUMNS)
    df["attack_class"] = df["label"].map(lambda x: ATTACK_CATEGORIES.get(x.strip().lower(), 0))
    df_feat = df.drop(["label", "difficulty_level", "attack_class"], axis=1)
    df_encoded = pd.get_dummies(df_feat, columns=["protocol_type", "service", "flag"])
    _engine.train_columns = df_encoded.columns

    # ── Scaler ──
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    with open(scaler_path, "rb") as f:
        _engine.scaler = pickle.load(f)

    # ── Sparse connectome ──
    connectome_path = os.path.join(BASE_DIR, "sparse_connectome.pt")
    if not os.path.exists(connectome_path):
        raise FileNotFoundError(f"Connectome not found: {connectome_path}")
    sparse_adj = torch.load(connectome_path, weights_only=False)
    num_neurons = sparse_adj.shape[0]

    # ── Model ──
    weights_path = os.path.join(BASE_DIR, "fly_brain_weights.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    n_features = len(_engine.train_columns)
    model = FlyBrainNet(n_features, num_neurons, 5, sparse_adj)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()
    _engine.model = model

    # ── Training metrics ──
    metrics_path = os.path.join(BASE_DIR, "training_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            _engine.metrics = json.load(f)
    else:
        _engine.metrics = {}


# ═══════════════════════════════════════════════════════════
# Feature Engineering (mirrors packet_engine.py logic)
# ═══════════════════════════════════════════════════════════
def _features_to_tensor(packet: PacketFeatures) -> np.ndarray:
    """
    One-hot encode categorical fields, align to training columns,
    and scale — identical pipeline to packet_engine.packet_to_model_input().
    """
    raw = packet.model_dump()

    df = pd.DataFrame([raw])
    df_encoded = pd.get_dummies(df, columns=["protocol_type", "service", "flag"])

    # Align to training columns: add missing, drop extra
    for col in _engine.train_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[_engine.train_columns]

    scaled = _engine.scaler.transform(df_encoded)
    return scaled[0]


# ═══════════════════════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load all artefacts on startup, print banner."""
    print("\n" + "=" * 60)
    print("  🧬 BIOLOGICAL FIREWALL ENGINE — REST API")
    print("  Powered by Drosophila melanogaster connectome")
    print("=" * 60)
    print("  Loading model artefacts …")
    _load_engine()
    n = _engine.model.num_neurons
    s = _engine.model.adjacency_mask._nnz()
    print(f"  ✓ Model loaded — {n:,} neurons · {s:,} synapses")
    print(f"  ✓ Scaler loaded — {len(_engine.train_columns):,} features")
    print(f"  🚀 API ready on port 8503")
    print("=" * 60 + "\n")
    yield


app = FastAPI(
    title="Biological Firewall Engine API",
    description=(
        "REST API for the fly-brain connectome cyber-threat classifier. "
        "Uses the *Drosophila melanogaster* connectome as a biologically-constrained "
        "neural network for 5-class network intrusion detection."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════
@app.post(
    "/api/v1/classify",
    response_model=ClassifyResponse,
    summary="Classify a network packet",
    description="Accept NSL-KDD features and return the connectome classification with confidence scores.",
)
async def classify_packet(packet: PacketFeatures) -> ClassifyResponse:
    """Run inference on a single packet's features."""
    if _engine.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        features = _features_to_tensor(packet)
        t_in = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred, _ = _engine.model(t_in)

        probs = torch.softmax(pred, dim=1)[0]
        cls_id = int(torch.argmax(probs).item())
        cls_name = CLASS_NAMES[cls_id]
        confidence = float(probs[cls_id].item()) * 100

        scores = [
            ClassScore(
                class_id=i,
                class_name=CLASS_NAMES[i],
                confidence=round(float(probs[i].item()) * 100, 4),
            )
            for i in range(len(CLASS_NAMES))
        ]

        return ClassifyResponse(
            classification=cls_name,
            class_id=cls_id,
            confidence=round(confidence, 4),
            verdict="BENIGN" if cls_id == 0 else "THREAT",
            scores=scores,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="Model health check",
    description="Returns key model statistics: neuron count, synapse count, accuracy, and F1 score.",
)
async def health_check() -> HealthResponse:
    """Return model health summary from training_metrics.json."""
    if _engine.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    metrics = _engine.metrics or {}
    return HealthResponse(
        status="ok",
        model="FlyBrainNet",
        num_neurons=metrics.get("num_neurons", _engine.model.num_neurons),
        num_synapses=metrics.get("num_synapses", _engine.model.adjacency_mask._nnz()),
        accuracy=round(metrics.get("best_val_accuracy", 0.0), 4),
        f1_macro=round(metrics.get("f1_macro", 0.0), 4),
    )


@app.get(
    "/api/v1/metrics",
    summary="Full training metrics",
    description="Returns the complete training_metrics.json content.",
)
async def training_metrics() -> Dict[str, Any]:
    """Return the raw training metrics dict."""
    if _engine.metrics is None:
        raise HTTPException(status_code=404, detail="Training metrics not available")
    return _engine.metrics
