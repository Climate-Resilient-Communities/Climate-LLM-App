## Step 4
import jsonlines
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import re
import time
import cohere
import numpy as np
import pinecone
from pinecone import Pinecone
from pinecone import ServerlessSpec
import tqdm
from FlagEmbedding import BGEM3FlagModel
import pickle
from tqdm.auto import tqdm
import json


# def check_listoflists(item):
#     """Return True if item is a list of lists"""
#     if isinstance(item, list):
#         return all(isinstance(subitem, list) for subitem in item)
#     return False

def flatten_list(list_of_lists):
    """Return a flattened list from a list of lists"""
    flattened_list = []
    for lst in list_of_lists:
        if isinstance(lst, list):
            for item in lst:
                flattened_list.append(item)
        else:
            flattened_list.append(lst)
    return flattened_list

def re_segment(text_lst: list, text_splitter, max_token_length: int) -> list:
    """Return resegmented text chunks from a list of texts"""
    comb_segment = " ".join(text_lst)
    segment_num_tokens = num_tokens_from_string(comb_segment, "cl100k_base")
    if segment_num_tokens < max_token_length:
      return [comb_segment]
    else:
      return text_splitter.split_text(comb_segment)


def vector_preprocessing(docs, text_splitter, max_token_length):
    """Return preprocessed metadata objects and text chunks.
    This allows the end user to see the content of what's being returned by their search instead of just the sparse/dense vectors"""
    metadata_objs = []
    text_chunks = []
    for doc in docs:
      title = doc['title']
      url = doc['url']
      segments = doc['metadata']
      doc_keywords = doc['keywords']

      for segment in segments:
        section_title = segment['section_title']
        segment_keywords = segment['keywords']
        segment_id = segment['id']
        table = segment['table']

        if type(segment_id) == list:
            segments = flatten_list(segment['segment'])
            chunk_text = re_segment(segments, text_splitter, max_token_length)
        else:
          chunk_text = segment['segment']

        if table != None:
          if table != "":
              chunk_text += ["".join(table)]

        for chunk in chunk_text:
          if not chunk.isspace():
              metadata_obj = {
                  "title": title,
                  "url": url,
                  "section_title": section_title,
                  "chunk_text": chunk,
                  "segment_id": str(segment_id),
                  "segment_keywords": segment_keywords,
                  "doc_keywords": doc_keywords
              }
              metadata_objs.append(metadata_obj)
              text_chunks.append(chunk)

    return metadata_objs, text_chunks

def get_embeddings(chunks, embed_model):
    embeddings = model.encode(sentences_1, return_dense=True, return_sparse=True, return_colbert_vecs=False)
    dense_embeddings = embeddings['dense_vecs']
    sparse_embeddings_lst = embeddings['lexical_weights']
    sparse_embeddings = []
    for sparse_embedding in sparse_embeddings_lst:
        sparse_dict = {}
        sparse_dict['indices'] = [int(index) for index in list(sparse_embedding.keys())]
        sparse_dict['values'] = list(sparse_embedding.values())
        sparse_embeddings.append(sparse_dict)
    return dense_embeddings, sparse_embeddings

def upsert_document(index, docs, batch_size, text_splitter, max_token_length, embed_model):
    metadata_objs, text_chunks = vector_preprocessing(docs, text_splitter, max_token_length)
    batch_error_lst = []
    id_error_lst = []
    for i in tqdm(range(0, len(text_chunks), batch_size)):
        # find end of batch
        i_end = min(i + batch_size, len(text_chunks))
        # create unique IDs
        ids = [str(x) for x in range(i, i_end)]
        # extract batch
        text_batch = text_chunks[i:i_end]
        metadata_batch = metadata_objs[i:i_end]
        # get dense and sparse embeddings
        dense_embeddings, sparse_embeddings = get_embeddings(text_batch, embed_model)

        vectors = []
        # loop through the data and create dictionaries for uploading documents to pinecone index
        for _id, sparse, dense, metadata in zip(ids, sparse_embeddings, dense_embeddings, metadata_batch):
            if sparse != {'indices': [], 'values': []} and dense != None:
                vectors.append({
                            'id': _id,
                            'sparse_values': sparse,
                            'values': dense,
                            'metadata': metadata
                        })
            else:
                 print(f"{_id}: Error getting chunk embeddings. Skipping storing chunk embeddings.")
                 print(metadata)
                 id_error_lst.append((_id, metadata))

        # upload the documents to the new hybrid index
        try:
            index.upsert(vectors=vectors)
        except Exception as e:
            print(e)
            for vector in vectors:
                try:
                  index.upsert(vectors=[vector])
                except Exception as e:
                  print(e)
                  batch_error_lst.append((vector['id'], vector['metadata']))
            continue
        time.sleep(5)
    return batch_error_lst, id_error_lst

if __name__ == "__main__":
    load_dotenv("YOUR SECRET ENVIRONMENT HERE, e.g. '../../secrets.env'")
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

    input_jsonl_path = sys.argv[1]
    finetune_embed_cd = sys.argv[2]
    finetune_embed_model_path = sys.argv[3]
    batch_size = sys.argv[4] #16
    erorr_dump_path = sys.argv[5]

    if finetune_embed_cd == False:
        embed_model = BGEM3FlagModel('BAAI/bge-m3',  
                       use_fp16=False) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    else:
        embed_model = pickle.load(open(finetune_embed_model_path, 'rb'))

    # Load documents
    docs = []
    with jsonlines.open(input_jsonl_path) as reader:
        for obj in reader:
            docs.append(obj)


    # Text spliter
    max_token_length = 1024
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1024, chunk_overlap=0
    )

    # Connect to Pinecone vector store 
    cloud = "aws"
    region = "us-east-1"
    pinecone_client = Pinecone(PINECONE_API_KEY)
    spec = ServerlessSpec(cloud=cloud, region=region)
    index_name = "climate-change-adaptation-index-10-24-prod"

    # if the index does not exist, we create it
    if index_name not in pinecone_client.list_indexes().names():
        pinecone_client.create_index(
            name=index_name,
            dimension=1024,
            metric="dotproduct",
            spec= spec
        )
        # wait for index to be initialized
        while not pinecone_client.describe_index(index_name).status['ready']:
            time.sleep(1)


    # Connect to index
    index = pinecone_client.Index(index_name) 

    # Upsert Vectors
    batch_error_lst, id_error_lst = upsert_document(index, docs, batch_size, text_splitter, max_token_length, embed_model)

    error_dict = {
        "batch_error_lst": batch_error_lst,
        "id_error_lst": id_error_lst
    }
    with open(error_dump_path, "w") as outfile: 
        json.dump(error_dict, outfile)
