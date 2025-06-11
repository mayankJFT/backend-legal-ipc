import time
import uuid
import json
import logging
from typing import AsyncGenerator
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from app.config import AVAILABLE_MODELS
from app.models import QueryRequest, QueryResponse, ResponseMetadata
from app.utils.prompts import final_prompt, fusion_prompt
from app.utils.helpers import (
    is_simple_greeting, get_greeting_response, format_docs, 
    count_tokens, format_conversation_history
)
from app.services.redis_service import redis_service
from app.services.vector_service import vector_service

logger = logging.getLogger("NyayaGPT-API")

class LLMService:
    def __init__(self):
        self.models = self._init_models()
    
    def _init_models(self):
        """Initialize LLM models configuration"""
        return {
            "gpt-4o": lambda streaming=False: ChatOpenAI(
                model="gpt-4o", 
                temperature=0.1, 
                max_tokens=1500,
                streaming=streaming,
                request_timeout=20
            ),
            "gpt-4o-mini": lambda streaming=False: ChatOpenAI(
                model="gpt-4o-mini", 
                temperature=0.1, 
                max_tokens=1500,
                streaming=streaming,
                request_timeout=15
            ),
            "gpt-3.5-turbo": lambda streaming=False: ChatOpenAI(
                model="gpt-3.5-turbo", 
                temperature=0.1, 
                max_tokens=1200,
                streaming=streaming,
                request_timeout=10
            )
        }
    
    def get_llm(self, model_name: str, streaming: bool = False):
        """Get LLM instance"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available. Available models: {list(self.models.keys())}")
        return self.models[model_name](streaming=streaming)
    
    def fusion_strategy(self, query, llm):
        """Optimized fusion strategy for faster retrieval"""
        try:
            vector_store = vector_service.get_vector_store()
            
            # Skip fusion for very short queries
            if len(query.split()) <= 3:
                return self.simple_strategy(query, llm)
                
            fusion_chain = fusion_prompt | llm
            response = fusion_chain.invoke({"question": query})
            variants = [line.strip("- ") for line in response.content.strip().split("\n") if line.strip()][:2]
            variants.insert(0, query)
            
            seen = set()
            all_docs = []
            
            # Retrieve fewer documents per variant for speed
            for variant in variants[:2]:  # Only use first 2 variants
                for doc in vector_store.similarity_search(variant, k=3):  # Reduced from 5 to 3
                    hash_ = doc.page_content[:50]  # Shorter hash for speed
                    if hash_ not in seen:
                        seen.add(hash_)
                        all_docs.append(doc)
            
            return all_docs[:3]  # Return max 3 documents
        except Exception as e:
            logger.warning(f"Fusion strategy failed, falling back to simple: {str(e)}")
            return self.simple_strategy(query, llm)

    def simple_strategy(self, query, llm):
        """Optimized direct retrieval"""
        vector_store = vector_service.get_vector_store()
        return vector_store.similarity_search(query, k=3)  # Reduced from 5 to 3
    
    async def process_query(self, query_request: QueryRequest):
        """Process a query with improved error handling and conversation management"""
        start_time = time.time()
        
        conversation_id = query_request.conversation_id or str(uuid.uuid4())
        
        try:
            user_message = {
                "role": "user",
                "content": query_request.query,
                "timestamp": time.time()
            }
            await redis_service.save_message_to_conversation(conversation_id, user_message)
            
            if not query_request.stream:
                cached = await redis_service.get_cached_response(
                    query_request.query, 
                    query_request.model_name,
                    query_request.strategy
                )
                if cached:
                    cached["metadata"]["conversation_id"] = conversation_id
                    
                    assistant_message = {
                        "role": "assistant",
                        "content": cached["response"],
                        "timestamp": time.time()
                    }
                    await redis_service.save_message_to_conversation(conversation_id, assistant_message)
                    
                    return QueryResponse(**cached)
            
            llm = self.get_llm(query_request.model_name, streaming=query_request.stream)
            llm.temperature = query_request.temperature
            llm.max_tokens = query_request.max_tokens
            
            conversation_history = ""
            if query_request.include_history:
                past_messages = await redis_service.get_conversation(conversation_id)
                if len(past_messages) > 1:
                    conversation_history = format_conversation_history(past_messages[:-1])
            
            if is_simple_greeting(query_request.query):
                greeting_response = get_greeting_response(query_request.query)
                
                assistant_message = {
                    "role": "assistant",
                    "content": greeting_response,
                    "timestamp": time.time()
                }
                await redis_service.save_message_to_conversation(conversation_id, assistant_message)
                
                duration = time.time() - start_time
                
                response = QueryResponse(
                    response=greeting_response,
                    metadata=ResponseMetadata(
                        model="fast-path-greeting",
                        strategy="direct",
                        chunks_retrieved=0,
                        tokens_used=0,
                        processing_time=round(duration, 2),
                        conversation_id=conversation_id
                    ),
                    context_sources=[]
                )
                
                return response
                
            retrieve_fn = self.fusion_strategy if query_request.strategy == "fusion" else self.simple_strategy
            
            try:
                docs = retrieve_fn(query_request.query, llm)
            except Exception as e:
                logger.warning(f"Error in retrieval: {str(e)}. Falling back to simple strategy.")
                docs = self.simple_strategy(query_request.query, llm)
                
            # Format documents and create context (optimized for speed)
            context = format_docs(docs, max_length=300)
            
            # Create prompt with history  
            prompt = final_prompt.format(
                history=conversation_history,
                context=context, 
                question=query_request.query
            )
            
            # Skip token counting for speed
            tokens_used = len(prompt) // 4
            
            parser = StrOutputParser()
            answer = (llm | parser).invoke(prompt)
            
            assistant_message = {
                "role": "assistant",
                "content": answer,
                "timestamp": time.time()
            }
            await redis_service.save_message_to_conversation(conversation_id, assistant_message)
            
            sources = [
                {
                    "title": doc.metadata.get("title", "Untitled"),
                    "url": doc.metadata.get("url", "No URL"),
                    "snippet": doc.page_content[:150] + "..."
                }
                for doc in docs
            ]
            
            duration = time.time() - start_time
            
            response = QueryResponse(
                response=answer,
                metadata=ResponseMetadata(
                    model=query_request.model_name,
                    strategy=query_request.strategy,
                    chunks_retrieved=len(docs),
                    tokens_used=tokens_used,
                    processing_time=round(duration, 2),
                    conversation_id=conversation_id
                ),
                context_sources=sources
            )
            
            if not query_request.stream:
                await redis_service.cache_response(
                    query_request.query,
                    query_request.model_name,
                    query_request.strategy,
                    response.dict()
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    async def generate_streaming_response(self, query_request: QueryRequest) -> AsyncGenerator[str, None]:
        """Generate a streaming response for the query with improved error handling."""
        start_time = time.time()
        
        conversation_id = query_request.conversation_id or str(uuid.uuid4())
        
        try:
            user_message = {
                "role": "user",
                "content": query_request.query,
                "timestamp": time.time()
            }
            await redis_service.save_message_to_conversation(conversation_id, user_message)
            
            llm = self.get_llm(query_request.model_name, streaming=True)
            llm.temperature = query_request.temperature
            llm.max_tokens = query_request.max_tokens
            
            conversation_history = ""
            if query_request.include_history:
                past_messages = await redis_service.get_conversation(conversation_id)
                if len(past_messages) > 1:
                    conversation_history = format_conversation_history(past_messages[:-1])
            
            if is_simple_greeting(query_request.query):
                greeting_response = get_greeting_response(query_request.query)
                
                yield f"data: {json.dumps({'chunk': greeting_response, 'full': greeting_response})}\n\n"
                
                assistant_message = {
                    "role": "assistant",
                    "content": greeting_response,
                    "timestamp": time.time()
                }
                await redis_service.save_message_to_conversation(conversation_id, assistant_message)
                
                duration = time.time() - start_time
                
                completion_data = {
                    "done": True,
                    "metadata": {
                        "model": "fast-path-greeting",
                        "strategy": "direct",
                        "chunks_retrieved": 0,
                        "tokens_used": 0,
                        "processing_time": round(duration, 2),
                        "conversation_id": conversation_id
                    },
                    "context_sources": []
                }
                
                yield f"data: {json.dumps(completion_data)}\n\n"
                return
                
            retrieve_fn = self.fusion_strategy if query_request.strategy == "fusion" else self.simple_strategy
            
            try:
                docs = retrieve_fn(query_request.query, llm)
            except Exception as e:
                logger.warning(f"Error in retrieval: {str(e)}. Falling back to simple strategy.")
                docs = self.simple_strategy(query_request.query, llm)
                
            context = format_docs(docs, max_length=600)
        
            prompt = final_prompt.format(
                history=conversation_history,
                context=context, 
                question=query_request.query
            )
            
            tokens_used = count_tokens(prompt, query_request.model_name)
            
            chain = llm | StrOutputParser()
            
            full_response = ""
            async for chunk in chain.astream(prompt):
                full_response += chunk
                yield f"data: {json.dumps({'chunk': chunk, 'full': full_response})}\n\n"
            
            assistant_message = {
                "role": "assistant",
                "content": full_response,
                "timestamp": time.time()
            }
            await redis_service.save_message_to_conversation(conversation_id, assistant_message)
            
            duration = time.time() - start_time
            
            sources = [
                {
                    "title": doc.metadata.get("title", "Untitled"),
                    "url": doc.metadata.get("url", "No URL"),
                    "snippet": doc.page_content[:150] + "..."
                }
                for doc in docs
            ]
            
            completion_data = {
                "done": True,
                "metadata": {
                    "model": query_request.model_name,
                    "strategy": query_request.strategy,
                    "chunks_retrieved": len(docs),
                    "tokens_used": tokens_used,
                    "processing_time": round(duration, 2),
                    "conversation_id": conversation_id
                },
                "context_sources": sources
            }
            
            yield f"data: {json.dumps(completion_data)}\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            error_data = {
                "error": str(e),
                "full": f"I apologize, but I encountered an error while processing your request. Please try again or contact support if the issue persists."
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            
            completion_data = {
                "done": True,
                "metadata": {
                    "model": query_request.model_name,
                    "strategy": query_request.strategy,
                    "chunks_retrieved": 0,
                    "tokens_used": 0,
                    "processing_time": round(time.time() - start_time, 2),
                    "conversation_id": conversation_id
                },
                "context_sources": [],
                "error": str(e)
            }
            
            yield f"data: {json.dumps(completion_data)}\n\n"

# Global LLM service instance
llm_service = LLMService()