
def rerank_fcn(query, docs_to_rerank, top_k):
    """Returns reranked documents that were retrieved"""
    rank_fields = ['title', 'section_title', 'content', 'segment_keywords', 'doc_keywords']
    rerank_results = cohere_client.rerank(
            query=query,
            documents=docs_to_rerank,
            top_n= top_k,
            model= "rerank-multilingual-v3.0",
            rank_fields=rank_fields
        )
    docs_retrieved = [docs_to_rerank[doc.index] for doc in rerank_results.results]
    return docs_retrieved

if __name__=="__main__":
    docs_to_rerank = sys.argv[1]
    query = sys.argv[2]

    load_dotenv("YOUR SECRET ENVIRONMENT HERE, e.g. '../../secrets.env'")
    COHERE_API_KEY = os.getenv('COHERE_API_KEY')
    cohere_client = cohere.Client(COHERE_API_KEY)

    rerank_fcn(query, docs_to_rerank, 10)
