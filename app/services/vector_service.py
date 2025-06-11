import logging
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from app.config import PINECONE_API_KEY, PINECONE_INDEX_NAME

logger = logging.getLogger("NyayaGPT-API")

class VectorService:
    def __init__(self):
        self.vector_store = None
    
    def init_vector_store(self):
        """Initialize Pinecone vector store with error handling"""
        try:
            # Initialize Pinecone
            if not PINECONE_API_KEY:
                logger.error("Pinecone API key not found")
                raise ValueError("Pinecone API key is required")
            
            pc = Pinecone(api_key=PINECONE_API_KEY)
            
            if not pc.has_index(PINECONE_INDEX_NAME):
                logger.error(f"Pinecone index '{PINECONE_INDEX_NAME}' does not exist")
                raise ValueError(f"Pinecone index '{PINECONE_INDEX_NAME}' does not exist")
            
            index = pc.Index(PINECONE_INDEX_NAME)
            
            self.vector_store = PineconeVectorStore(
                index=index, 
                embedding=OpenAIEmbeddings(model="text-embedding-ada-002")
            )
            
            logger.info("Vector store initialized")
            return self.vector_store
        except Exception as e:
            logger.error(f"Vector store initialization error: {str(e)}")
            raise
    
    def get_vector_store(self):
        """Get the vector store instance"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        return self.vector_store

# Global vector service instance
vector_service = VectorService()