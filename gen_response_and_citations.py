import os 
import cohere

def doc_preprocessing(docs):
    """Returns preprocessed documents so that Cohere model can cite relevant ones"""
    documents = []
    for doc in docs:
        document = {}
        title = doc['title']
        url = doc['url']
        chunk_text = doc['chunk_text']
        if isinstance(url, list) and len(url) >= 1:
            document['data'] = {
                "title": title + ": " + url[0],
                "snippet": chunk_text
            }
        else:
            document['data'] = {
                "title": title,
                "snippet": chunk_text
            }
        documents.append(document)
    return documents

def cohere_chat(query, documents):
    """Returns Command R Plus model response and citations"""
    documents_processed = doc_preprocessing(documents)
    res = co.chat(
        model="command-r-plus-08-2024",
        messages=[
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": f"Question: {query}. \n Answer:",
        },
    ],
        documents= documents_processed)
    return res.message.content[0].text, res.message.citations

if __name__=="__main__":
    docs_reranked = sys.argv[1]
    query = sys.argv[2]

    load_dotenv("YOUR SECRET ENVIRONMENT HERE, e.g. '../../secrets.env'")
    COHERE_API_KEY = os.getenv('COHERE_API_KEY')
    co = cohere.ClientV2(api_key=COHERE_API_KEY)

    system_message="""
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
    """

    cohere_chat(query, docs_reranked)
