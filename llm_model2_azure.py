# -*- coding: utf-8 -*-
import asyncio
import os
from dotenv import load_dotenv
import warnings
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_cohere import ChatCohere
from langchain.schema import Document
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph
import dask
from dask.distributed import Client
from azure.storage.blob import BlobServiceClient
import os
import shutil

# Load environment variables from .env file
load_dotenv('secrets.env')

# Check for required environment variables
required_env_vars = ['COHERE_API_KEY', 'LANGCHAIN_TRACING_V2', 'LANGCHAIN_ENDPOINT', 'LANGCHAIN_API_KEY', 'TAVILY_API_KEY']
missing_vars = [var for var in required_env_vars if os.getenv(var) is None]

if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Set environment variables
os.environ['COHERE_API_KEY'] = os.getenv('COHERE_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ['LANGCHAIN_ENDPOINT'] = os.getenv('LANGCHAIN_ENDPOINT')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

# Suppress all warnings of type Warning (superclass of all warnings)
warnings.filterwarnings("ignore", category=Warning)

# Initialize Dask client inside main guard
if __name__ == "__main__":
    client = Client(n_workers=4, threads_per_worker=2, memory_limit='2GB')

def download_blob_to_local_file(container_name, blob_name, download_file_path):
    # Get connection string from environment variable
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # Create a blob client
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    # Download the blob to a local file
    with open(download_file_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())

def download_embeddings_folder(container_name, local_folder_path):
    # Get connection string from environment variable
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # Get the container client
    container_client = blob_service_client.get_container_client(container_name)

    # Ensure local directory exists
    if os.path.exists(local_folder_path):
        shutil.rmtree(local_folder_path)  # Clear the directory if it exists
    os.makedirs(local_folder_path)

    # List and download blobs
    blob_list = container_client.list_blobs()
    for blob in blob_list:
        # Define the path where the blob will be downloaded
        blob_download_path = os.path.join(local_folder_path, blob.name)
        # Create directories if they do not exist
        os.makedirs(os.path.dirname(blob_download_path), exist_ok=True)
        # Download the blob
        download_blob_to_local_file(container_name, blob.name, blob_download_path)

# Set embeddings
def load_embeddings():
    container_name = "embeddings"
    local_folder_path = "./embeddings"

    # Download the embeddings folder from Azure Blob Storage
    download_embeddings_folder(container_name, local_folder_path)

    # Set embeddings
    embd = CohereEmbeddings()

    # Load the vectorstore from the downloaded embeddings folder
    loaded_vectorstore = Chroma(
        persist_directory=local_folder_path,
        embedding_function=embd
    )

    # Create a retriever from the loaded_vectorstore
    retriever = loaded_vectorstore.as_retriever()

    return retriever

def get_embeddings():
    return load_embeddings()

# Load PDF names from pdffiles.txt
with open("pdfiles.txt", "r") as file:
    pdf_files = [line.strip() for line in file]

pdf_files_str = ", ".join(pdf_files)

# Data models
class QuestionVerification(BaseModel):
    """Checks if the question is related to climate change or global warming topics."""
    query: str = Field(description="The query to evaluate.")

class web_search(BaseModel):
    query: str = Field(description="The query to use when searching the internet.")

class vectorstore(BaseModel):
    query: str = Field(description="The query to use when searching the vectorstore.")

# Preamble
preamble = f"""
You are an intelligent assistant trained to first evaluate if a user's question is saying hi or introducing themselves then if the content relates in any way to climate change topics or the environment and if not, do not answer.
If the question is related to climate change, decide whether to use the vectorstore containing documents on {pdf_files_str}, or to route climate change topics or global warming question to general web search.
"""

# Set up the LLM with the ability to make routing decisions
llm = ChatCohere(model="command-r-plus", temperature=0)
structured_llm_router = llm.bind_tools(tools=[QuestionVerification, web_search, vectorstore], preamble=preamble)

# Define a prompt that asks the LLM to make a routing decision
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", preamble),
        ("human", "{question}"),
        ("system", "Based on the question content, should this query be directed to the vectorstore for climate-specific documents, or should it be searched on the web for other information?")
    ]
)

# Combine the route prompt with the LLM routing logic
question_router = route_prompt | structured_llm_router

### Retrieval Grader

# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

# Prompt
preamble = """You are a grader assessing relevance of a retrieved document to a user question. \n
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

# LLM with function call
llm = ChatCohere(model="command-r-plus", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments, preamble=preamble)

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

### Generate

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
import langchain
from langchain_core.messages import HumanMessage

# Preamble
preamble = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. You need to be friendly, factual and can use up to 1 paragraph and bullet points to explain thoroughly."""

# LLM
llm = ChatCohere(model_name="command-r-plus", temperature=0).bind(preamble=preamble)

# Prompt
prompt = lambda x: ChatPromptTemplate.from_messages(
    [
        HumanMessage(
            f"Question: {x['question']} \nAnswer: ",
            additional_kwargs={"documents": x["documents"]},
        )
    ]
)

# Chain
rag_chain = prompt | llm | StrOutputParser()

# Preamble for concise answers
preamble = """You are an assistant for question-answering tasks. Answer the question based upon your knowledge. Use three sentences maximum and keep the answer concise."""

# LLM for concise answers
llm = ChatCohere(model_name="command-r-plus", temperature=0).bind(preamble=preamble)

# Prompt for concise answers
prompt = lambda x: ChatPromptTemplate.from_messages(
    [
        HumanMessage(
            f"Question: {x['question']} \nAnswer: "
        )
    ]
)

# Chain for concise answers
llm_chain = prompt | llm | StrOutputParser()

### Hallucination Grader

# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

# Preamble
preamble = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

# LLM with function call
llm = ChatCohere(model="command-r-plus", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations, preamble=preamble)

# Prompt
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader

### Answer Grader

# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

# Preamble
preamble = """You are a grader assessing whether an answer addresses / resolves a question \n
Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

# LLM with function call
llm = ChatCohere(model="command-r-plus", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer, preamble=preamble)

# Prompt
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader

### Search
from langchain_community.tools.tavily_search import TavilySearchResults
web_search_tool = TavilySearchResults()

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

# Use Dask for parallel retrieval
@dask.delayed
def fetch_vectorstore_documents(question):
    return retriever.invoke(question)

@dask.delayed
def fetch_web_search_results(query):
    return web_search_tool.invoke({"query": query})

@dask.delayed
def fetch_llm_response(question):
    return llm_chain.invoke({"question": question})

async def retrieve(state):
    question = state["question"]
    vectorstore_docs = fetch_vectorstore_documents(question)
    web_search_results = fetch_web_search_results(question)
    llm_response = fetch_llm_response(question)
    
    delayed_results = [vectorstore_docs, web_search_results, llm_response]
    results = dask.compute(*delayed_results)
    return {"documents": results[0], "question": question}

async def llm_fallback(state):
    question = state["question"]
    llm_response = await fetch_llm_response(question)
    return {"question": question, "generation": llm_response}

async def rag(state):
    question = state["question"]
    documents = state["documents"]
    if not isinstance(documents, list):
        documents = [documents]
    generation = await rag_chain.ainvoke({"documents": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

async def grade_documents(state):
    question = state["question"]
    documents = state["documents"]

    tasks = [retrieval_grader.ainvoke({"question": question, "document": d.page_content}) for d in documents]
    scores = await asyncio.gather(*tasks)

    filtered_docs = [d for d, score in zip(documents, scores) if score.binary_score == "yes"]
    return {"documents": filtered_docs, "question": question}

async def verify_question(state):
    question = state["question"]

    if question.lower().startswith(("hello", "hi", "how are you", "are you okay", "how's your day")):
        return "llm_fallback"

    verification = await question_router.ainvoke({"question": question})

    if "tool_calls" in verification.response_metadata and len(verification.response_metadata["tool_calls"]) > 0:
        return "retrieve"
    else:
        return "not_related"

async def not_related_response(state):
    question = state["question"]
    generation = "Sorry, this question is not related to climate change. Do you have any questions related to the topic?"
    return {"question": question, "generation": generation}

async def web_search(state):
    question = state["question"]
    docs = await web_search_tool.ainvoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    return {"documents": web_results, "question": question}

### Edges ###

async def route_question(state):
    next_node = await verify_question(state)
    if next_node is None:
        return END
    return next_node

async def decide_to_generate(state):
    filtered_documents = state["documents"]

    if not filtered_documents:
        return "web_search"
    else:
        return "rag"

async def grade_generation_v_documents_and_question(state):
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = await hallucination_grader.ainvoke({"documents": documents, "generation": generation})
    if score is None:
        return generation

    grade = score.binary_score

    if grade == "yes":
        score = await answer_grader.ainvoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            return "useful"
        else:
            return "not useful"
    else:
        return "not supported"

async def generate(state):
    question = state["question"]
    documents = state["documents"]
    if not isinstance(documents, list):
        documents = [documents]
    generation = state['generation']
    return {"documents": documents, "question": question, "generation": generation}

workflow = StateGraph(GraphState)

workflow.add_node("web_search", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("rag", rag)
workflow.add_node("llm_fallback", llm_fallback)
workflow.add_node("not_related_response", not_related_response)
workflow.add_node("generate", generate)

workflow.set_conditional_entry_point(
    route_question,
    {
        "retrieve": "retrieve",
        "not_related": "not_related_response",
        "llm_fallback": "llm_fallback",
    },
)
workflow.add_edge("llm_fallback", "generate")
workflow.add_edge("not_related_response", "generate")
workflow.add_edge("web_search", "rag")
workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "web_search": "web_search",
        "rag": "rag",
    },
)
workflow.add_conditional_edges(
    "rag",
    grade_generation_v_documents_and_question,
    {
        "not supported": "web_search",
        "not useful": "web_search",
        "useful": "generate"
    },
)
workflow.add_edge("generate", END)

app = workflow.compile()

# Helper function to run the workflow asynchronously
async def run_workflow(input_state):
    result = await app.ainvoke(input_state)
    return result
