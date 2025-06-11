import uvicorn
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import HOST, PORT, WORKERS, logger
from app.routes import router
from app.services.redis_service import redis_service
from app.services.vector_service import vector_service

# === Custom Lifespan Context Manager ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize services
    try:
        # Initialize Redis
        await redis_service.init_redis()
        logger.info("Redis connection established")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {str(e)}")
        # Continue without Redis - features requiring Redis will be disabled
    
    try:
        # Initialize vector store
        vector_service.init_vector_store()
        logger.info("Vector store initialized")
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        raise  # This is critical, so we should fail startup
    
    yield
    
    # Shutdown: Clean up resources
    await redis_service.close()

# === Initialize App ===
app = FastAPI(
    title="NyayaGPT API",
    description="Legal Assistant API powered by LLMs with RAG",
    version="1.0.0",
    lifespan=lifespan
)

# === Add CORS Middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Include Routes ===
app.include_router(router)

# === Server startup configuration ===
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        workers=WORKERS,
        reload=False,  # Set to False for production
        access_log=True,
        log_level="info"
    )