from pydantic import BaseModel

class ModelRequest(BaseModel):
    prompt: str

class ModelResponse(BaseModel):
    response: str