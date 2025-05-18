from api.v1.schemas.chat import ModelRequest, ModelResponse
from fastapi import APIRouter, HTTPException
import requests

chat = APIRouter(prefix="/chat", tags=["Chat"])

model_endpoint = ""

@chat.post("/generate", response_model=ModelResponse)
async def generate_response(request: ModelRequest):
    try:
        payload = {"inputs": request.prompt}
        # Assuming `model_endpoint` is set to the correct model URL
        response = requests.post(model_endpoint, json=payload)

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Model server error")

        model_response = response.json()
        # Adjust parsing depending on your model's response structure
        generated_text = (
            model_response[0]["generated_text"]
            if isinstance(model_response, list)
            else model_response["generated_text"]
        )

        return ModelResponse(response=generated_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
