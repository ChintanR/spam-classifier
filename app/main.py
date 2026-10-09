from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, field_validator

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "spam_model.joblib"
ml = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml["model"] = joblib.load(MODEL_PATH)   # load once at startup, not per request
    yield
    ml.clear()


app = FastAPI(title="Spam Classifier API", lifespan=lifespan)


class Message(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty")
        return v


class Prediction(BaseModel):
    label: str
    is_spam: bool
    spam_probability: float


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(msg: Message):
    prob = float(ml["model"].predict_proba([msg.text])[0][1])
    is_spam = prob >= 0.5
    return Prediction(
        label="spam" if is_spam else "ham",
        is_spam=is_spam,
        spam_probability=round(prob, 4),
    )