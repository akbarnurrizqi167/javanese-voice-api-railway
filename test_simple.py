"""
Simple test API without heavy dependencies
"""
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

app = FastAPI(title="Javanese API Test", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Javanese Voice API Test",
        "status": "running",
        "platform": "Railway Test"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "dependencies": "minimal"}

@app.post("/test-upload")
async def test_upload(file: UploadFile = File(...)):
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_kb": len(await file.read()) / 1024,
        "status": "upload_successful"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
