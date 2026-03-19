from fastapi import FastAPI, HTTPException, Query
from app.schemas import GenerateRequest, GenerateResponse
from app.service import generate_problems

app = FastAPI(
    title="Math Problem Generator API",
    version="1.0",
    description="API for generating math problems with LLM"
)


@app.get("/")
def root():
    return {"message": "Math Problem Generator API is running"}


@app.post("/generate", response_model=GenerateResponse)
def generate(
    request: GenerateRequest,
    temperature: float = Query(0.4, ge=0.0, le=2.0)
):
    try:
        result = generate_problems(request, temperature)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    