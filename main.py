import os
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from ollama import Client

load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "https://localhost:11434")

client=Client(host=OLLAMA_HOST)
app = FastAPI(title="Ollama FastAPI")

Role = Literal["system", "user", "assistant"]

class Message(BaseModel):
    role:Role
    content:str

class ChatRequest(BaseModel):
    model: str = "gemma2:2b"
    messages:list[Message]

class ChatResponse(BaseModel):
    response:str
    model: str="gemma2:2b"

#API Route
@app.post("/chat",response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = client.chat(
            model=request.model,
            messages=[m.model_dump() for m in request.messages],
            stream=False
        )
        return ChatResponse(
            response=result["message"]["content"],
            model=result.get("model", request.model),
        )
    except Exception as e:
        return ChatResponse(response=f"[error] {e}", model=request.model)

