from pydantic import BaseModel


class InvokeRequest(BaseModel):
    prompt: str


class InvokeResponse(BaseModel):
    response: str
