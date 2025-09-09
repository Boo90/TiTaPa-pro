
import os
import time
import json
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from passlib.hash import bcrypt
import jwt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pickle

JWT_SECRET = os.getenv("JWT_SECRET", "change-this-secret")
APP_DIR = os.getenv("APP_DIR", "/app")
DATA_DIR = os.path.join(APP_DIR, "data")
MODELS_DIR = os.path.join(APP_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(APP_DIR, 'db.sqlite3')}")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default="user")
    created_at = Column(DateTime, default=datetime.utcnow)

class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    filename = Column(String, nullable=False)
    path = Column(String, nullable=False)
    rows = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    owner = relationship("User")

class ModelRecord(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    target = Column(String, nullable=False)
    algo = Column(String, default="LogReg")
    path = Column(String, nullable=False)
    metrics = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow)
    owner = relationship("User")
    dataset = relationship("Dataset")

Base.metadata.create_all(bind=engine)

app = FastAPI(title="TiTaPa API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_token(user: User):
    payload = {"sub": user.email, "uid": user.id, "role": user.role, "exp": datetime.utcnow() + timedelta(hours=12)}
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def get_current_user(token: HTTPAuthorizationCredentials = Depends(security), db=Depends(get_db)) -> User:
    try:
        payload = jwt.decode(token.credentials, JWT_SECRET, algorithms=["HS256"])
        email = payload.get("sub")
        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")

class AuthRequest(BaseModel):
    email: str
    password: str

class AuthResponse(BaseModel):
    token: str
    email: str
    role: str

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

@app.post("/auth/register", response_model=AuthResponse)
def register(req: AuthRequest, db=Depends(get_db)):
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(email=req.email, password_hash=bcrypt.hash(req.password), role="admin" if req.email.endswith("@admin") else "user")
    db.add(user)
    db.commit()
    db.refresh(user)
    return AuthResponse(token=create_token(user), email=user.email, role=user.role)

@app.post("/auth/login", response_model=AuthResponse)
def login(req: AuthRequest, db=Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not bcrypt.verify(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return AuthResponse(token=create_token(user), email=user.email, role=user.role)

@app.post("/data/upload")
def upload(file: UploadFile = File(...), user: User = Depends(get_current_user), db=Depends(get_db)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".csv", ".txt"]:
        raise HTTPException(status_code=400, detail="Only CSV supported for now")
    ts = int(time.time())
    safe_name = f"{user.id}_{ts}_{file.filename}"
    path = os.path.join(DATA_DIR, safe_name)
    with open(path, "wb") as f:
        f.write(file.file.read())
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot parse CSV: {e}")
    ds = Dataset(owner_id=user.id, filename=file.filename, path=path, rows=len(df))
    db.add(ds)
    db.commit()
    db.refresh(ds)
    cols = list(df.columns)
    return {"dataset_id": ds.id, "rows": ds.rows, "columns": cols}

@app.get("/datasets")
def list_datasets(user: User = Depends(get_current_user), db=Depends(get_db)):
    q = db.query(Dataset).filter(Dataset.owner_id == user.id).order_by(Dataset.created_at.desc()).all()
    return [{"id": d.id, "filename": d.filename, "rows": d.rows, "created_at": d.created_at.isoformat()} for d in q]

class TrainRequest(BaseModel):
    dataset_id: int
    target: str
    test_size: float = 0.2
    random_state: int = 42

@app.post("/ml/train")
def train(req: TrainRequest, user: User = Depends(get_current_user), db=Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == req.dataset_id, Dataset.owner_id == user.id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = pd.read_csv(ds.path)
    if req.target not in df.columns:
        raise HTTPException(status_code=400, detail="Target column not found in dataset")
    X = df.drop(columns=[req.target]).select_dtypes(include=["number"])
    y = df[req.target]
    if X.empty:
        raise HTTPException(status_code=400, detail="No numeric features to train on")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=req.test_size, random_state=req.random_state, stratify=y if y.nunique() < 20 else None)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    model_path = os.path.join(MODELS_DIR, f"model_{user.id}_{int(time.time())}.pkl")
    import pickle
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "features": list(X.columns), "target": req.target}, f)
    rec = ModelRecord(owner_id=user.id, dataset_id=ds.id, target=req.target, algo="LogReg", path=model_path, metrics=json.dumps({"accuracy": acc}))
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return {"model_id": rec.id, "accuracy": acc}

@app.post("/ml/predict-json/{model_id}")
async def predict_json(model_id: int, body: dict, user: User = Depends(get_current_user), db=Depends(get_db)):
    rec = db.query(ModelRecord).filter(ModelRecord.id == model_id, ModelRecord.owner_id == user.id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="Model not found")
    import pickle
    with open(rec.path, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    features = bundle["features"]
    row = {k: body.get(k, 0) for k in features}
    import pandas as pd
    X = pd.DataFrame([row])[features].values
    pred = model.predict(X)[0]
    return {"prediction": str(pred), "used_features": features}

@app.get("/dashboards/kpis")
def kpis(user: User = Depends(get_current_user), db=Depends(get_db)):
    users = db.query(User).count()
    datasets = db.query(Dataset).filter(Dataset.owner_id == user.id).count()
    models = db.query(ModelRecord).filter(ModelRecord.owner_id == user.id).count()
    return {"users": users, "datasets": datasets, "models": models}
