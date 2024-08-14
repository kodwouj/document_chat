"""
Conversation Chain Module

This module sets up a conversational retrieval chain using the LangChain library
and a HuggingFace model. It provides functionality to create a conversation chain
that can answer questions based on a given vector store of document embeddings.
"""

import asyncio
import os
import logging
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from cachetools import TTLCache
from ratelimit import limits, sleep_and_retry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

CACHE_SIZE = 100
CACHE_TTL = 3600
cache = TTLCache(maxsize=CACHE_SIZE, ttl=CACHE_TTL)

RATE_LIMIT = 5
RATE_LIMIT_PERIOD = 60

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
@sleep_and_retry
@limits(calls=RATE_LIMIT, period=RATE_LIMIT_PERIOD)
async def get_conversation_chain(vectorstore):
    """
    Create a conversational chain that includes source metadata in the responses.
    """
    if not api_token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in the environment variables.")
    if not vectorstore:
        raise ValueError("A valid vector store must be provided.")

    vectorstore = await vectorstore

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.5, "max_length": 512},
        huggingfacehub_api_token=api_token
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True
    )

    return chain

async def ask_question(conversation_chain, query: str):
    """
    Ask a question using the conversation chain and get the answer with sources.
    """
    result = await asyncio.to_thread(conversation_chain, {'question': query})
    
    answer = result['answer']
    source_documents = result.get('source_documents', [])

    sources = []
    for doc in source_documents:
        metadata = doc.metadata
        source = metadata.get('source', 'Unknown')
        sources.append(source)
    
    return answer, sources
