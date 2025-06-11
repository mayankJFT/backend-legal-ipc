import re
import time
import random
import tiktoken
import logging
from typing import List, Dict

logger = logging.getLogger("NyayaGPT-API")

def is_simple_greeting(text):
    """Detect if input is a simple greeting that doesn't need RAG"""
    text = text.lower().strip()
    greeting_patterns = [
        r'^(hi|hello|hey|greetings|namaste|howdy)[\s\W]*$',
        r'^(good\s*(morning|afternoon|evening|day))[\s\W]*$',
        r'^(how\s*(are\s*you|is\s*it\s*going|are\s*things))[\s\W]*$',
        r'^(what\'*s\s*up)[\s\W]*$'
    ]
    
    for pattern in greeting_patterns:
        if re.match(pattern, text):
            return True
    return False

def get_greeting_response(greeting_text):
    """Generate appropriate response for simple greetings without using LLM"""
    greeting_text = greeting_text.lower().strip()
    
    if re.match(r'^(hi|hello|hey|howdy)[\s\W]*$', greeting_text):
        responses = [
            "Hello! How can I help you with legal information today?",
            "Hi there! I'm NyayaGPT, your legal assistant. What legal questions can I help you with?",
            "Hello! I'm ready to assist with your legal queries."
        ]
        return random.choice(responses)
    
    elif re.match(r'^(good\s*morning)[\s\W]*$', greeting_text):
        return "Good morning! How can I assist you with legal matters today?"
    
    elif re.match(r'^(good\s*afternoon)[\s\W]*$', greeting_text):
        return "Good afternoon! What legal questions can I help you with today?"
    
    elif re.match(r'^(good\s*evening)[\s\W]*$', greeting_text):
        return "Good evening! I'm here to help with any legal queries you might have."
    
    elif re.match(r'^(how\s*are\s*you)[\s\W]*$', greeting_text):
        return "I'm functioning well, thank you for asking! I'm ready to assist with your legal questions."
    
    elif re.match(r'^(what\'*s\s*up)[\s\W]*$', greeting_text):
        return "I'm here and ready to help with your legal queries! What can I assist you with today?"
    
    return "Hello! I'm NyayaGPT, your legal assistant. How can I help you today?"

def format_docs(docs, max_length=400):
    """Format documents with shorter length limit for faster processing"""
    result = []
    for doc in docs[:3]:  # Only use top 3 documents for speed
        title = doc.metadata.get("title", "Untitled Document")
        url = doc.metadata.get("url", "No URL")
        result.append(f"### {title}\n**Source:** {url}\n\n{doc.page_content.strip()[:max_length]}...")
    return "\n\n".join(result)

def count_tokens(text, model="gpt-3.5-turbo"):
    """Count tokens in text with error handling"""
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception as e:
        logger.warning(f"Error counting tokens: {str(e)}. Using approximate count.")
        return len(text) // 4

def format_conversation_history(messages, max_tokens=500):
    """Format conversation history with reduced token limit for speed"""
    formatted_history = []
    for msg in messages[-4:]:  # Only keep last 4 messages for speed
        role = msg.get("role", "user" if "query" in msg else "assistant")
        content = msg.get("content", msg.get("query", msg.get("response", "")))
        # Truncate long messages
        if len(content) > 200:
            content = content[:200] + "..."
        formatted_history.append(f"{role.capitalize()}: {content}")
    
    history_text = "\n\n".join(formatted_history)
    
    # Quick token estimation and truncation
    if len(history_text) > max_tokens * 4:  # Rough estimation
        history_text = history_text[-(max_tokens * 4):]
        history_text = "...\n" + history_text
    
    return history_text