import uuid
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi_limiter.depends import RateLimiter

from app.models import QueryRequest, HealthResponse
from app.config import AVAILABLE_MODELS
from app.services.llm_service import llm_service
from app.services.redis_service import redis_service

logger = logging.getLogger("NyayaGPT-API")

router = APIRouter()

# Rate limiter dependency with fallback
async def rate_limit_dependency():
    """Rate limiting dependency that works with or without Redis"""
    if redis_service.client:
        try:
            # Use rate limiter only if Redis is available
            limiter = RateLimiter(times=30, seconds=60)  # Increased limit
            await limiter()
        except Exception as e:
            logger.warning(f"Rate limiting failed: {str(e)}")
            # Continue without rate limiting
            pass

async def get_or_create_conversation(request: Request) -> str:
    """Get existing conversation ID from cookie or create a new one"""
    conversation_id = request.cookies.get("conversation_id")
    
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
        logger.info(f"Created new conversation: {conversation_id}")
    
    return conversation_id

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and available models"""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        available_models=AVAILABLE_MODELS
    )

@router.get("/")
async def root():
    """Root endpoint for health checks"""
    return {"message": "NyayaGPT API is running", "version": "1.0.0"}

@router.get("/status")
async def status():
    """Detailed status endpoint for monitoring"""
    status_info = {
        "api": "running",
        "redis": "disconnected",
        "vector_store": "disconnected",
        "timestamp": datetime.now().isoformat()
    }
    
    # Check Redis connection
    if redis_service.client:
        try:
            await redis_service.client.ping()
            status_info["redis"] = "connected"
        except Exception:
            status_info["redis"] = "error"
    
    # Check vector store
    try:
        vector_store = llm_service.simple_strategy("test", llm_service.get_llm("gpt-4o-mini"))
        status_info["vector_store"] = "connected"
    except Exception:
        status_info["vector_store"] = "error"
    
    return status_info

@router.post("/query")
async def query_endpoint(
    query_request: QueryRequest,
    background_tasks: BackgroundTasks,
    request: Request
):
    """Process a legal query using the specified LLM and retrieval strategy"""
    if not query_request.conversation_id:
        query_request.conversation_id = await get_or_create_conversation(request)
    
    if query_request.stream:
        response = StreamingResponse(
            llm_service.generate_streaming_response(query_request),
            media_type="text/event-stream"
        )
        
        response.set_cookie(
            key="conversation_id",
            value=query_request.conversation_id,
            httponly=True,
            max_age=30*24*60*60
        )
        
        return response
    
    try:
        response_data = await llm_service.process_query(query_request)
        
        response = JSONResponse(content=response_data.dict())
        response.set_cookie(
            key="conversation_id",
            value=query_request.conversation_id,
            httponly=True,
            max_age=30*24*60*60
        )
        
        return response
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in query endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

@router.get("/conversation/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """Retrieve conversation history by ID"""
    conversation = await redis_service.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation with ID {conversation_id} not found"
        )
    return {"conversation_id": conversation_id, "messages": conversation}

@router.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation by ID"""
    try:
        deleted = await redis_service.delete_conversation(conversation_id)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation with ID {conversation_id} not found"
            )
        
        return {"status": "success", "message": f"Conversation {conversation_id} deleted"}
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting conversation: {str(e)}"
        )

@router.get("/clear-cache")
async def clear_cache():
    """Clear the response cache"""
    try:
        deleted_count = await redis_service.clear_cache()
        return {
            "status": "success",
            "message": f"Cache cleared: {deleted_count} entries removed"
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing cache: {str(e)}"
        )