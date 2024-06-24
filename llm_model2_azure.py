# -*- coding: utf-8 -*-
import os
import warnings
import pickle
import json
import asyncio
import numpy as np
import shutil
from dotenv import load_dotenv
from typing import List
from azure.storage.blob import BlobServiceClient
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_cohere import ChatCohere
from langchain.schema import Document
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph
from dask.distributed import Client
import cohere
from pinecone import Pinecone, ServerlessSpec # type: ignore
from pinecone_text.sparse import BM25Encoder # type: ignore

# Load environment variables from .env file
load_dotenv('secrets.env')

# Check for required environment variables
required_env_vars = [
    'COHERE_API_KEY', 'LANGCHAIN_TRACING_V2', 'LANGCHAIN_ENDPOINT', 'LANGCHAIN_API_KEY',
    'TAVILY_API_KEY', 'AZURE_STORAGE_CONNECTION_STRING', 'PINECONE_API_KEY'
]
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

# Initialize Cohere client
cohere_api_key = os.getenv('COHERE_API_KEY')
cohere_client = cohere.Client(api_key=cohere_api_key)

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_client = Pinecone(api_key=pinecone_api_key)
index_name = "climate-change-adaptation-index"
spec = ServerlessSpec(cloud='aws', region='us-east1')

# Check if the index exists, if not, create it
if index_name not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        name=index_name,
        dimension=1024,
        metric="dotproduct",
        spec=spec
    )
index = pinecone_client.Index(index_name)

# Azure Blob Storage Utilities
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

def download_reference_folder(container_name, local_folder_path):
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

# Load data from Azure Blob Storage
def load_data():
    container_name = "reference"
    local_folder_path = "./reference"

    # Download the reference folder from Azure Blob Storage
    download_reference_folder(container_name, local_folder_path)

    # Load data
    with open(os.path.join(local_folder_path, 'bm25_encoder_model.pkl'), 'rb') as file:
        bm25 = pickle.load(file)

    with open(os.path.join(local_folder_path, 'markdown_chunked_docs.json')) as json_file:
        markdown_chunked_docs = json.load(json_file)

    with open(os.path.join(local_folder_path, 'pdfiles.txt'), 'r') as file:
        pdf_files = [line.strip() for line in file]

    return bm25, markdown_chunked_docs, pdf_files

# Load data from Azure Blob Storage
bm25, markdown_chunked_docs, pdf_files = load_data()
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
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

### Generate

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

# Preamble
preamble="""
You are an expert in climate change and global warming. You will be answering questions from a broad audience that includes high school students and professionals. You should adopt the persona of an educator, providing information that is both accessible and engaging.

Persona:
Consider yourself an educator for both youth and adults.
Ensure your responses are helpful, harmless, and honest.

Language:
Easy to read and understand for grade 9 students.

Tone and Style:
Friendly and approachable
Free of jargon
Factual and accurate

Content Requirements:
Detailed and complete responses
Use bullet points for clarity
Provide intuitive examples when possible

Leverage Constitutional AI:
Align your responses with human values.
Ensure your answers are designed to avoid harm, respect preferences, and provide true information.

Example Question: What is climate change and why should we care?
Response:
Let's talk about climate change and why it matters to all of us.

**What is Climate Change?**

- **Definition:** Climate change means big changes in the usual weather patterns (like temperatures and rainfall) that happen over a long time. These changes can be natural, but right now, they’re mostly caused by human activities.
- **Key Factors:**

  - **Greenhouse Gases (GHGs):** When we burn fossil fuels (like coal, oil, and natural gas) for energy, it releases gases that trap heat in the atmosphere.

  - **Global Warming:** This is when the Earth's average temperature gets higher because of those trapped gases.

**Why Should We Care?**

- **Impact on Weather:**

  - **Extreme Weather Events:** More frequent and intense heatwaves, hurricanes, and heavy rainstorms can lead to serious damage and danger.
  - **Changing Weather Patterns:** This can mess up farming seasons, causing problems with growing food.

- **Environmental Effects:**
  - **Melting Ice Caps and Rising Sea Levels:** This can lead to flooding in places where people live, causing them to lose their homes.
  - **Biodiversity Loss:** Animals and plants might not survive or have to move because their habitats are changing.

- **Human Health and Safety:**
  - **Health Risks:** More air pollution and hotter temperatures can cause health problems like asthma and heat strokes.
  - **Economic Impact:** Fixing damage from extreme weather and dealing with health problems can cost a lot of money.

**What Can We Do to Help?**

- **Reduce Carbon Footprint:**

  - **Energy Efficiency:** Use devices that save energy, like LED bulbs and efficient appliances.
  - **Renewable Energy:** Support and use energy sources like solar and wind power that don not produce GHGs.

- **Adopt Sustainable Practices:**

  - **Reduce, Reuse, Recycle:** Cut down on waste by following these three steps.
  - **Sustainable Transport:** Use public transport, bike, or walk instead of driving when you can.
**Why Your Actions Matter:**

- **Collective Impact:** When lots of people make small changes, it adds up to a big positive effect on our planet.
- **Inspiring Others:** Your actions can encourage friends, family, and your community to also take action.
**Let's Make a Difference Together!**

  - **Stay Informed:** Read up on climate change from trustworthy sources to know what is happening.
  - **Get Involved:** Join local or online groups that work on climate action.
  
**Questions or Curious About Something?**

Feel free to ask any questions or share your thoughts. We are all in this together, and every little bit helps!
"""

# LLM
llm = ChatCohere(model_name="command-r-plus", temperature=0).bind(preamble=preamble)

# Prompt
prompt = lambda x: ChatPromptTemplate.from_messages(
    [
        HumanMessage(
            f"Question: {x['question']} \nAnswer: ",
            additional_kwargs={"documents": x["documents"]}
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
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
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
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")
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
    citations: str

async def fetch_web_search_results(query):
    return await web_search_tool.ainvoke({"query": query})

async def fetch_llm_response(question):
    return await llm_chain.ainvoke({"question": question})

async def retrieve(state):
    question = state["question"]

    # Perform hybrid search to get initial set of documents
    reranked_docs = rerank_fcn(question, bm25, alpha=0.3, top_k=5)
    documented_docs = [Document(page_content=docs['section_text'], metadata={'filename': docs['header']}) for docs in reranked_docs]

    return {"documents": documented_docs, "question": question}

async def llm_fallback(state):
    question = state["question"]
    llm_response = await fetch_llm_response(question)
    return {"question": question, "generation": llm_response}

async def rag(state):
    question = state["question"]
    documents = state["documents"]
    citations = state["citations"]

    if not isinstance(documents, list):
        documents = [documents]
    
    generation = await rag_chain.ainvoke({"documents": documents, "question": question})
    return {"documents": documents, "citations":citations, "question": question, "generation": generation}

async def grade_documents(state):
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    for d in documents:
        score = await retrieval_grader.ainvoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade == "yes":
            filtered_docs.append(d)
    citations = "  \n".join(set(doc.metadata.get('filename') for doc in filtered_docs))
    return {"documents": filtered_docs, "citations": citations, "question": question}

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

def web_search(state):
    question = state["question"]
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    citations = "  \n".join([d["url"] for d in docs])
    return {"documents": web_results, "citations": citations, "question": question}

# Hybrid Search and Re-ranking

def get_dense_embeddings_query(query):
    try:
        response = cohere_client.embed(
            model='embed-english-v3.0',
            input_type="search_query",
            texts=[query],
            truncate='END'
        )
        return np.asarray(response.embeddings)
    except Exception as e:
        print(f"Error getting embeddings: {str(e)}")
        return None

def weight_by_alpha(sparse_embedding, dense_embedding, alpha):
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    hsparse = {
        'indices': sparse_embedding['indices'],
        'values': [v * (1 - alpha) for v in sparse_embedding['values']]
    }
    hdense = [v * alpha for v in dense_embedding]
    return hsparse, hdense

def get_hybrid_results(query, bm25, alpha, top_k):
    query_sparse_embedding = bm25.encode_queries(query)
    query_dense_embedding = get_dense_embeddings_query(query)
    try:
        scaled_sparse, scaled_dense = weight_by_alpha(query_sparse_embedding, query_dense_embedding[0], alpha)
        hybrid = index.query(
            vector=scaled_dense,
            sparse_vector=scaled_sparse,
            top_k=top_k,
            include_metadata=True
        )
        return hybrid
    except Exception as e:
        print(f"Error querying index: {str(e)}")
        return None

def rerank_fcn(query, bm25, alpha, top_k):
    try:
        hybrid = get_hybrid_results(query, bm25, alpha, top_k)
        if hybrid is None or "matches" not in hybrid:
            print("Error: No matches found in hybrid results.")
            return []

        docs_to_rerank = [doc.get('metadata') for doc in hybrid.get("matches")]
    
        # Filter out documents with only empty or whitespace-only fields
        valid_docs_to_rerank = []
        for doc in docs_to_rerank:
            if doc:
                valid_doc = {field: value for field, value in doc.items() if field != 'order_id' and value and value.strip()}
                if valid_doc:
                    valid_docs_to_rerank.append(valid_doc)

        if not valid_docs_to_rerank:
            print("No valid documents to rerank.")
            return []

        # Prepare the documents in the expected format for Cohere Rerank API
        formatted_docs = [{"text": " ".join(doc.values())} for doc in valid_docs_to_rerank]

        rerank_results = cohere_client.rerank(
            query=query,
            documents=formatted_docs,
            top_n=top_k,
            model="rerank-english-v3.0"
        )

        docs_retrieved = [valid_docs_to_rerank[doc.index] for doc in rerank_results.results]
        
        for doc in docs_retrieved:
            section_splits_list = markdown_chunked_docs[doc['header']][doc['section_header']]
            section_text = " "
            for split in section_splits_list:
                section_text = "".join(split[1])
                doc['section_text'] = section_text

        return docs_retrieved
    
    except Exception as e:
        print(f"Error in reranking: {str(e)}")
        return []

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
    citations = state["citations"]

    if not isinstance(documents, list):
        documents = [documents]
    
    return {"documents": documents, "citations":citations, "question": question, "generation": state['generation']}

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
    }
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
    }
)
workflow.add_conditional_edges(
    "rag",
    grade_generation_v_documents_and_question,
    {
        "not supported": "web_search",
        "not useful": "web_search",
        "useful": "generate"
    }
)
workflow.add_edge("generate", END)

app = workflow.compile()

# Helper function to run the workflow asynchronously
async def run_workflow(input_state):
    result = await app.ainvoke(input_state)
    return result
