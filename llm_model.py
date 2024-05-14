# -*- coding: utf-8 -*-
### LLMs
import os
from langchain_core._api.beta_decorator import LangChainBetaWarning
from dotenv import load_dotenv
import warnings

# Load environment variables from .env file
load_dotenv('secrets.env')

# Suppress all warnings of type Warning (superclass of all warnings)
warnings.filterwarnings("ignore", category=Warning)
warnings.simplefilter("ignore", LangChainBetaWarning)

# Set up Cohere client and choose model
os.environ['COHERE_API_KEY'] = os.getenv('COHERE_API_KEY')

# Tracing Optional
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ['LANGCHAIN_ENDPOINT'] = os.getenv('LANGCHAIN_ENDPOINT')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

# Search
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata

####  ONLY RUN TO TO DO EMBEDDINGS ON PDFS
# # Set embeddings
# embd = CohereEmbeddings()

# # Path to the folder containing PDF files
# pdf_folder = "Big_PDF"

# # Get the list of PDF files in the folder
# pdf_files = [file for file in os.listdir(pdf_folder) if file.endswith(".pdf")]

# docs = []

# for pdf_file in pdf_files:
#     pdf_path = os.path.join(pdf_folder, pdf_file)
#     loader = UnstructuredPDFLoader(file_path=pdf_path, mode="elements")
#     document_elements = loader.load()
#     docs.extend(document_elements)
#     print(f"Loaded {len(document_elements)} elements from {pdf_file}")

# # Clean complex metadata from documents
# docs = filter_complex_metadata(docs, allowed_types=(str, bool, int, float))

# # Split text into manageable chunks for better indexing
# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=512, chunk_overlap=0
# )
# doc_splits = text_splitter.split_documents(docs)
# print(f"Total document splits: {len(doc_splits)}") ### Added in to print

# embeddings_dir = "embeddings"
# os.makedirs(embeddings_dir, exist_ok=True)

# # Load the existing vectorstore if it exists, otherwise create a new one
# if os.path.exists(os.path.join(embeddings_dir, "index")):
#     vectorstore = Chroma(
#         persist_directory=embeddings_dir,
#         embedding_function=embd
#     )
# else:
#     vectorstore = Chroma.from_documents(
#         documents=doc_splits,
#         embedding=embd,
#         persist_directory=embeddings_dir
#     )

# # Define the maximum batch size
# max_batch_size = 5000

# # Split doc_splits into batches
# batches = [doc_splits[i:i+max_batch_size] for i in range(0, len(doc_splits), max_batch_size)]

# # Add documents to the vectorstore in batches
# for batch in batches:
#     vectorstore.add_documents(batch)
#     print(f"Added {len(batch)} documents to the vectorstore")
#####----------------------------------
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings

# Set embeddings
embd = CohereEmbeddings()

# Path to the embeddings folder
embeddings_dir = "embeddings"

# Load the vectorstore from the embeddings folder
loaded_vectorstore = Chroma(
    persist_directory=embeddings_dir,
    embedding_function=embd
)

# Create a retriever from the loaded vectorstore
retriever = loaded_vectorstore.as_retriever()

### Router
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_cohere import ChatCohere

# Load PDF names from pdffiles.txt
with open("pdfiles.txt", "r") as file:
    pdf_files = [line.strip() for line in file]
# Join the PDF files with commas for the preamble
pdf_files_str = ", ".join(pdf_files)

# Data models
class QuestionVerification(BaseModel):
    """Checks if the question is related to climate change or global warming topics."""
    query: str = Field(description="The query to evaluate.")

class web_search(BaseModel):
   f"""
   The internet. Use web_search for questions that are related to anything else than {pdf_files_str}.
   """
   query: str = Field(description="The query to use when searching the internet.")

class vectorstore(BaseModel):
    f"""
    A vectorstore containing documents related to {pdf_files_str}. Use the vectorstore for questions on these topics.
    """
    query: str = Field(description="The query to use when searching the vectorstore.")

# Preamble ***NOTE PLEASE MODIFY HERE THE NAMES OF THE PDFS YOU USE OR IT WONT SEND TO THE RIGHT ACTIONS***
preamble = f"""
You are an intelligent assistant trained to first evaluate if a user's question is saying hi or introducing themselves then if the content relates in any way to climate change topics or the enviroment and if not, do not answer.
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


from langchain import hub
from langchain_core.output_parsers import StrOutputParser
import langchain
from langchain_core.messages import HumanMessage


# Preamble
preamble = """You are an assistant for question-answering tasks. Answer the question based upon your knowledge. Use three sentences maximum and keep the answer concise."""

# LLM
llm = ChatCohere(model_name="command-r-plus", temperature=0).bind(preamble=preamble)

# Prompt
prompt = lambda x: ChatPromptTemplate.from_messages(
    [
        HumanMessage(
            f"Question: {x['question']} \nAnswer: "
        )
    ]
)

# Chain
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
        # ("system", system),
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


from typing_extensions import TypedDict
from typing import List

class GraphState(TypedDict):
    """|
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """
    question : str
    generation : str
    documents : List[str]

from langchain.schema import Document

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def llm_fallback(state):
    """
    Generate answer using the LLM w/o vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---LLM Fallback---")
    question = state["question"]
    generation = llm_chain.invoke({"question": question})
    
    return {"question": question, "generation": generation}

def rag(state):
    """
    Generate answer using the documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    if not isinstance(documents, list):
      documents = [documents]

    # RAG generation
    generation = rag_chain.invoke({"documents": documents, "question": question})
    
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}

def verify_question(state):
    """
    Verify if the question is related to climate change or global warming topics, or a welcome or introduction.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call based on the verification result, or "not_related" if the question is not related to climate change
    """

    print("---VERIFY QUESTION---")
    question = state["question"]

    # If the question is a personal question directly asking the LLM, route it to the LLM for generating a response
    if question.lower().startswith(("hello", "hi", "how are you", "are you okay", "how's your day")):
        print("---PERSONAL QUESTION, ROUTE TO LLM---")
        return "llm_fallback"

    verification = question_router.invoke({"question": question})

    if "tool_calls" in verification.response_metadata and len(verification.response_metadata["tool_calls"]) > 0:
        print("---QUESTION IS RELATED TO CLIMATE CHANGE, ROUTE TO VECTORSTORE---")
        return "retrieve"
    else:
        print("---QUESTION IS NOT RELATED TO CLIMATE CHANGE---")
        return "not_related"

def not_related_response(state):
    """
    Respond to questions not related to climate change.

    Args:
        state (dict): The current graph state

    Returns:
        str: A message indicating that the question is not related to climate change
    """
    question = state["question"]
    generation = "Sorry, this question is not related to climate change. Do you have any questions related to the topic?"
    return {"question": question ,"generation": generation}

def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    
    return {"documents": web_results, "question": question}

### Edges ###

def route_question(state):
    """
    Route question based on the verification result.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    #Maryam
    #or None if the question is not related to climate change

    print("---ROUTE QUESTION---")
    next_node = verify_question(state)
    if next_node is None:
        return END
    return next_node

    # Fallback to LLM or raise error if no decision
    if "tool_calls" not in source.additional_kwargs:
        print("---ROUTE QUESTION TO LLM---")
        return "llm_fallback"
    if len(source.additional_kwargs["tool_calls"]) == 0:
      raise "Router could not decide source"

    # Choose datasource
    datasource = source.additional_kwargs["tool_calls"][0]["function"]["name"]
    if datasource == 'web_search':
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif datasource == 'vectorstore':
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    else:
        print("---ROUTE QUESTION TO LLM---")
        return "vectorstore"

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    #question = state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, WEB SEARCH---")
        return "web_search"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "rag"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers the question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call or the generation if hallucination grading fails
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    if score is None:
        print("Hallucination grading failed to return a result. Showing unverified generation.")
        return generation  # Returning the generation directly if hallucination check fails

    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint.pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

def generate(state):
    """
    Return genaration

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): The current graph state including question, documents and generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    if not isinstance(documents, list):
      documents = [documents]

    generation = state['generation']
    
    return {"documents": documents, "question": question, "generation": generation}

import pprint

from langgraph.graph import END, StateGraph

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("web_search", web_search) # web search
workflow.add_node("retrieve", retrieve) # retrieve
workflow.add_node("grade_documents", grade_documents) # grade documents
workflow.add_node("rag", rag) # rag
workflow.add_node("llm_fallback", llm_fallback) # llm
workflow.add_node("not_related_response", not_related_response) # not related
workflow.add_node("generate",generate)

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
        "not supported": "web_search", # Hallucinations: re-generate
        "not useful": "web_search", # Fails to answer question: fall-back to web-search
        "useful": "generate"
    },
)
workflow.add_edge("generate", END)

app = workflow.compile()
