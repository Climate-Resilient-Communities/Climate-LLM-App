
import os
import re
import time
import cohere
import pinecone
from pinecone import Pinecone
from FlagEmbedding import BGEM3FlagModel


def get_query_embeddings(query, embed_model):
    embeddings = model.encode([query], return_dense=True, return_sparse=True, return_colbert_vecs=False)
    query_dense_embeddings = embeddings['dense_vecs']
    query_sparse_embeddings_lst = embeddings['lexical_weights']
    query_sparse_embeddings = []
    for sparse_embedding in query_sparse_embeddings_lst:
        sparse_dict = {}
        sparse_dict['indices'] = [int(index) for index in list(sparse_embedding.keys())]
        sparse_dict['values'] = list(sparse_embedding.values())
        query_sparse_embeddings.append(sparse_dict)
    return query_dense_embeddings, query_sparse_embeddings

def weight_by_alpha(sparse_embedding, dense_embedding, alpha):
    """
    Weight the values of our sparse and dense embeddings by the parameter alpha (0-1).

    :param sparse_embedding: Sparse embedding representation of one of our documents (or chunks).
    :param dense_embedding: Dense embedding representation of one of our documents (or chunks).
    :param alpha: Weighting parameter between 0-1 that controls the impact of sparse or dense embeddings on the retrieval and ranking
        of returned docs (chunks) in our index.

    :return: Weighted sparse and dense embeddings for one of our documents (chunks).
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    hsparse = {
        'indices': sparse_embedding['indices'],
        'values':  [v * (1 - alpha) for v in sparse_embedding['values']]
    }
    hdense = [v * alpha for v in dense_embedding]
    return hsparse, hdense

def issue_hybrid_query(index, sparse_embedding, dense_embedding, alpha, top_k):
    """
    Send properly formatted hybrid search query to Pinecone index and get back `k` ranked results (ranked by dot product similarity, as
        defined when we made our index).

    :param sparse_embedding: Sparse embedding representation of one of our documents (or chunks).
    :param dense_embedding: Dense embedding representation of one of our documents (or chunks).
    :param alpha: Weighting parameter between 0-1 that controls the impact of sparse or dense embeddings on the retrieval and ranking
        of returned docs (chunks) in our index.
    :param top_k: The number of documents (chunks) we want back from Pinecone.

    :return: QueryResponse object from Pinecone containing top-k results.
    """
    scaled_sparse, scaled_dense = weight_by_alpha(sparse_embedding, dense_embedding, alpha)

    result = index.query(
        vector=scaled_dense,
        sparse_vector=scaled_sparse,
        top_k=top_k,
        include_metadata=True
    )
    return result

def get_hybrid_results(index, query, embed_model, alpha, top_k):
    query_dense_embeddings, query_sparse_embeddings = get_query_embeddings(query, embed_model)
    """
    pure_keyword = issue_hybrid_query(query_sembedding, query_dembedding[0], 1.0, 5)
    pure_semantic = issue_hybrid_query(query_sembedding, query_dembedding[0], 0.0, 5)
    hybrid_1 = issue_hybrid_query(query_sembedding, query_dembedding[0], 0.1, 5)
    hybrid_2 = issue_hybrid_query(query_sembedding, query_dembedding[0], 0.2, 5)
    hybrid_3 = issue_hybrid_query(query_sembedding, query_dembedding[0], 0.3, 5)
    hybrid_4 = issue_hybrid_query(query_sembedding, query_dembedding[0], 0.4, 5)
    hybrid_5 = issue_hybrid_query(query_sembedding, query_dembedding[0], 0.5, 5)
    """
    hybrid = issue_hybrid_query(index, query_sparse_embedding, query_dense_embedding[0], alpha, top_k)
    return hybrid

def get_entire_content(original_docs, docs_to_rerank):
    docs_to_rerank_new = []
    for doc in docs_to_rerank:
        title = doc['title']
        segment_id = eval(doc['segment_id'])
        matched_doc = [doc for doc in original_docs if doc['title'] == title]
        matched_content = [metadata['content'] for metadata in matched_doc['metadata'] if metadata['id'] == segment_id][0]
        doc['content'] = matched_content
        docs_to_rerank_new.append(doc)
    return docs_to_rerank_new

if __name__=="__main__":
    load_dotenv("YOUR SECRET ENVIRONMENT HERE, e.g. '../../secrets.env'")
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    pinecone_client = Pinecone(PINECONE_API_KEY)

    index_name = sys.argv[1]
    input_jsonl_path = sys.argv[2]
    finetune_embed_cd = sys.argv[3]
    finetune_embed_model_path = sys.argv[4]
    query = sys.argv[5]

    # Load documents
    original_docs = []
    with jsonlines.open(input_jsonl_path) as reader:
        for obj in reader:
            original_docs.append(obj)

    # Connect to index
    index = pinecone_client.Index(index_name)

    # Load embedding model 
    if finetune_embed_cd == False:
        embed_model = BGEM3FlagModel('BAAI/bge-m3',  
                       use_fp16=False) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    else:
        embed_model = pickle.load(open(finetune_embed_model_path, 'rb'))

    # Perform vector search 
    hybrid = get_hybrid_results(index, query, embed_model, 0.4, 10)
    docs_to_rerank = [i.get('metadata') for i in hybrid.get("matches")]


    # Retrieve entire section of text instead of a small chunk
    get_entire_content(original_docs, docs_to_rerank)
