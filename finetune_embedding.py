
import json
import jsonlines
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from llama_index.core import Document
import os
import re
import time
import cohere
import pinecone
from pinecone import Pinecone
from llama_index.llms.cohere import Cohere
from llama_index.finetuning import generate_qa_embedding_pairs
from llama_index.finetuning import SentenceTransformersFinetuneEngine
import pickle


def load_corpus(docs, verbose=False):
    """Load summary corpus from docs into llama_index Documents"""
    summary_lst = []
    for doc in docs:
        summary = doc['intro_conclusion'] + doc['summary']
        summary_lst.append(summary)

    docs = [Document(text = summary) for summary in summary_lst]
    if verbose:
        print(f"Loaded {len(docs)} docs")

    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

    if verbose:
        print(f"Parsed {len(nodes)} nodes")

    return nodes


if __name__ == "__main__":

    load_dotenv("YOUR SECRET ENVIRONMENT HERE, e.g. '../../secrets.env'")
    COHERE_API_KEY = os.getenv('COHERE_API_KEY')
    cohere_llm = Cohere(model="command-r-plus", api_key=COHERE_API_KEY)
    

    input_jsonl_path = sys.argv[1]
    output_train_dataset_json_path = sys.arg[2]
    output_finetuned_embedding_model_pkl_path = sys.argv[3]


    # Load documents
    docs = []
    with jsonlines.open(input_jsonl_path) as reader:
        for obj in reader:
            docs.append(obj)  

    # Construct training dataset
    train_nodes = load_corpus(docs, verbose = True)
    val_nodes = train_nodes  

    train_dataset = generate_qa_embedding_pairs(
        llm=cohere_llm, nodes=train_nodes
    )

    val_dataset = generate_qa_embedding_pairs(
        llm=cohere_llm, nodes=val_nodes
    )

    train_dataset.save_json(output_train_dataset_json_path)

    # Finetune Multilingual Embedding Model 
    finetune_engine = SentenceTransformersFinetuneEngine(
        train_dataset,
        model_id='BAAI/bge-m3', #"intfloat/multilingual-e5-large" 
        model_output_path="finetuned_model",
        val_dataset=val_dataset,
    )
    finetune_engine.finetune()
    embed_model = finetune_engine.get_finetuned_model()
    with open(output_finetuned_embedding_model_pkl_path, 'wb') as file:
        pickle.dump(embed_model, file)
