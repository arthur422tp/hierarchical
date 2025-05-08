from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import uvicorn
import os

# Get the absolute path of the app directory
APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_DIR, "static")

app = FastAPI(title="RAG System API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class QueryRequest(BaseModel):
    query: str
    use_extraction: bool = False

class QueryResponse(BaseModel):
    answer: str
    retrieved_docs: List[str]

@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        index_path = os.path.join(STATIC_DIR, "index.html")
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Static files not found. Please ensure the app/static directory exists with index.html")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading static files: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        # TODO: Implement your RAG logic here
        # This is a placeholder response
        return QueryResponse(
            answer="This is a placeholder response",
            retrieved_docs=["Document 1", "Document 2"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Ensure the static directory exists
    if not os.path.exists(STATIC_DIR):
        os.makedirs(STATIC_DIR)
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 