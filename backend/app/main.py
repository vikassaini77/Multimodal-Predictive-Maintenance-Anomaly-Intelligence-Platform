from fastapi import FastAPI
from backend.app.api.router import router as graph_router

app = FastAPI(
    title="Multimodal Predictive Maintenance API",
    description="API for evaluating GraphSAGE fault propagation across factory topology.",
    version="1.0.0"
)

app.include_router(graph_router, prefix="/graph", tags=["graph"])

@app.get("/health")
def health_check():
    return {"status": "healthy"}
