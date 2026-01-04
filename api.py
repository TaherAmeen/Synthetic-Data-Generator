"""
FastAPI-based REST API for the Synthetic Review Data Generator.

This API provides endpoints for generating synthetic reviews
and managing the generation process.
"""

import json
import os
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Import the core generation logic
from app import load_config, run_generation, create_run_folder, DEFAULT_CONFIG_PATH, OUTPUT_DIR, REPORTS_DIR

# Store for tracking background jobs
jobs: Dict[str, Dict[str, Any]] = {}


# Pydantic models for request/response
class Product(BaseModel):
    name: str
    type: str
    description: str


class Persona(BaseModel):
    role: str
    description: str


class ModelConfig(BaseModel):
    provider: str = Field(..., description="Model provider: openai, ollama, or mistral")
    name: str = Field(..., description="Model name, e.g., gpt-4o-mini")


class GenerationOptions(BaseModel):
    pros_and_cons: bool = True
    rating: bool = True
    use_research: bool = False


class GenerationConfig(BaseModel):
    products: List[Product]
    personas: List[Persona]
    rating_distribution: Dict[str, float] = Field(
        default={"1": 0.1, "2": 0.1, "3": 0.2, "4": 0.3, "5": 0.3}
    )
    models: List[ModelConfig]
    samples_number: int = Field(default=10, ge=1, le=1000)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    options: GenerationOptions = Field(default_factory=GenerationOptions)

    class Config:
        json_schema_extra = {
            "example": {
                "products": [
                    {
                        "name": "Easygenerator",
                        "type": "E-learning platform",
                        "description": "Cloud-based e-learning platform for creating online courses."
                    }
                ],
                "personas": [
                    {
                        "role": "Instructional Designer",
                        "description": "Focuses on course creation and learner engagement."
                    }
                ],
                "rating_distribution": {"1": 0.1, "2": 0.1, "3": 0.2, "4": 0.3, "5": 0.3},
                "models": [{"provider": "openai", "name": "gpt-4o-mini"}],
                "samples_number": 10,
                "similarity_threshold": 0.7,
                "options": {
                    "pros_and_cons": True,
                    "rating": True,
                    "use_research": False
                }
            }
        }


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    reports_folder: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class GenerationResponse(BaseModel):
    job_id: str
    status: str
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Ensure output directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    yield


# Create FastAPI app
app = FastAPI(
    title="Synthetic Review Generator API",
    description="""
    API for generating synthetic product reviews using LLMs.
    
    ## Features
    - Generate synthetic reviews with configurable personas and products
    - Support for multiple LLM providers (OpenAI, Ollama, Mistral)
    - Background job processing for long-running generations
    - Quality assurance checks (similarity, sentiment, reality)
    - Comprehensive reporting
    
    ## Usage
    1. Start a generation job with POST /generate
    2. Check job status with GET /jobs/{job_id}
    3. Download results when complete
    """,
    version="1.0.0",
    lifespan=lifespan
)


def run_generation_job(job_id: str, config: dict):
    """Background task to run the generation process."""
    jobs[job_id]["status"] = "running"
    jobs[job_id]["started_at"] = datetime.now().isoformat()
    
    # Create a run-specific folder for this job
    run_folder = create_run_folder()
    jobs[job_id]["reports_folder"] = run_folder
    
    try:
        result = run_generation(config, reports_dir=run_folder)
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
        # Serialize result (convert to JSON-serializable format)
        serialized_result = {}
        for model_id, model_data in result.items():
            serialized_result[model_id] = {
                "output_file": model_data.get("output_file"),
                "reviews_count": len(model_data.get("reviews", [])),
                "reports_folder": run_folder
            }
        jobs[job_id]["result"] = serialized_result
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        jobs[job_id]["error"] = str(e)


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Synthetic Review Generator API",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "output_dir_exists": os.path.exists(OUTPUT_DIR),
        "reports_dir_exists": os.path.exists(REPORTS_DIR),
        "active_jobs": len([j for j in jobs.values() if j["status"] == "running"])
    }


@app.post("/generate", response_model=GenerationResponse, tags=["Generation"])
async def start_generation(
    config: GenerationConfig,
    background_tasks: BackgroundTasks
):
    """
    Start a synthetic review generation job.
    
    The generation runs in the background. Use the returned job_id
    to check status and retrieve results.
    """
    job_id = str(uuid.uuid4())
    
    # Store job info
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "reports_folder": None,
        "error": None,
        "result": None,
        "config": config.model_dump()
    }
    
    # Convert Pydantic model to dict for processing
    config_dict = config.model_dump()
    
    # Schedule background task
    background_tasks.add_task(run_generation_job, job_id, config_dict)
    
    return GenerationResponse(
        job_id=job_id,
        status="pending",
        message="Generation job started. Use GET /jobs/{job_id} to check status."
    )


@app.post("/generate/sync", tags=["Generation"])
async def generate_sync(config: GenerationConfig):
    """
    Run generation synchronously (blocking).
    
    Warning: This can take a long time for large sample sizes.
    Consider using the async /generate endpoint for production use.
    """
    try:
        config_dict = config.model_dump()
        run_folder = create_run_folder()
        result = run_generation(config_dict, reports_dir=run_folder)
        
        # Serialize result
        serialized_result = {}
        for model_id, model_data in result.items():
            serialized_result[model_id] = {
                "output_file": model_data.get("output_file"),
                "reviews_count": len(model_data.get("reviews", [])),
                "reviews": model_data.get("reviews", []),
                "reports_folder": run_folder
            }
        
        return {
            "status": "completed",
            "reports_folder": run_folder,
            "result": serialized_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/generate/default", response_model=GenerationResponse, tags=["Generation"])
async def start_generation_with_default_config(background_tasks: BackgroundTasks):
    """
    Start generation using the default config.json file.
    """
    try:
        config = load_config(DEFAULT_CONFIG_PATH)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load default config: {e}")
    
    job_id = str(uuid.uuid4())
    
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "reports_folder": None,
        "error": None,
        "result": None,
        "config": config
    }
    
    background_tasks.add_task(run_generation_job, job_id, config)
    
    return GenerationResponse(
        job_id=job_id,
        status="pending",
        message="Generation job started with default config. Use GET /jobs/{job_id} to check status."
    )


@app.get("/jobs", response_model=List[JobStatus], tags=["Jobs"])
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(10, ge=1, le=100)
):
    """List all generation jobs."""
    job_list = list(jobs.values())
    
    if status:
        job_list = [j for j in job_list if j["status"] == status]
    
    # Sort by created_at descending
    job_list.sort(key=lambda x: x["created_at"], reverse=True)
    
    return job_list[:limit]


@app.get("/jobs/{job_id}", response_model=JobStatus, tags=["Jobs"])
async def get_job_status(job_id: str):
    """Get the status of a specific generation job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]


@app.delete("/jobs/{job_id}", tags=["Jobs"])
async def delete_job(job_id: str):
    """Delete a job from the tracking list."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if jobs[job_id]["status"] == "running":
        raise HTTPException(status_code=400, detail="Cannot delete a running job")
    
    del jobs[job_id]
    return {"message": f"Job {job_id} deleted"}


@app.get("/config/default", tags=["Configuration"])
async def get_default_config():
    """Get the default configuration."""
    try:
        config = load_config(DEFAULT_CONFIG_PATH)
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load config: {e}")


@app.get("/outputs", tags=["Outputs"])
async def list_outputs():
    """List all generated output files."""
    if not os.path.exists(OUTPUT_DIR):
        return {"files": []}
    
    files = []
    for filename in os.listdir(OUTPUT_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(OUTPUT_DIR, filename)
            files.append({
                "filename": filename,
                "size_bytes": os.path.getsize(filepath),
                "created_at": datetime.fromtimestamp(os.path.getctime(filepath)).isoformat()
            })
    
    files.sort(key=lambda x: x["created_at"], reverse=True)
    return {"files": files}


@app.get("/outputs/{filename}", tags=["Outputs"])
async def get_output_file(filename: str):
    """Download a specific output file."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(filepath, media_type="application/json", filename=filename)


@app.get("/outputs/{filename}/content", tags=["Outputs"])
async def get_output_content(filename: str):
    """Get the content of an output file as JSON."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    
    with open(filepath, "r") as f:
        content = json.load(f)
    
    return content


@app.get("/reports", tags=["Reports"])
async def list_reports():
    """List all generated reports."""
    if not os.path.exists(REPORTS_DIR):
        return {"files": []}
    
    files = []
    for filename in os.listdir(REPORTS_DIR):
        if filename.endswith((".json", ".md")):
            filepath = os.path.join(REPORTS_DIR, filename)
            files.append({
                "filename": filename,
                "type": "json" if filename.endswith(".json") else "markdown",
                "size_bytes": os.path.getsize(filepath),
                "created_at": datetime.fromtimestamp(os.path.getctime(filepath)).isoformat()
            })
    
    files.sort(key=lambda x: x["created_at"], reverse=True)
    return {"files": files}


@app.get("/reports/{filename}", tags=["Reports"])
async def get_report_file(filename: str):
    """Download a specific report file."""
    filepath = os.path.join(REPORTS_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    
    media_type = "application/json" if filename.endswith(".json") else "text/markdown"
    return FileResponse(filepath, media_type=media_type, filename=filename)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
