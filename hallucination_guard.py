from datasets import Dataset
import os
from ragas import evaluate
from dotenv import load_dotenv
from ragas.database_schema import SingleTurnSample 
from ragas.metrics import FaithfulnesswithHHEM

def extract_contexts(citations, docs_reranked):
  contexts = []
  if citations != None:
      for citation in citations:
        sources = citation.sources
        for source in sources:
          document = source.document
          context = document.get('title') + ": " + document.get('snippet')
          contexts.append(context)
  else:
      for document in docs_retrieved:
        context = document.get('title') + ": " + document.get('chunk_text')
        contexts.append(context)
  return contexts

  
if __name__=="__main__":
    question = sys.argv[1]
    answer = sys.argv[2]
    docs_reranked = sys.argv[3]
    citations = sys.argv[4]

    load_dotenv("YOUR SECRET ENVIRONMENT HERE, e.g. '../../secrets.env'")
    COHERE_API_KEY = os.getenv('COHERE_API_KEY')
    os.environ["COHERE_API_KEY"] = COHERE_API_KEY

    from ragas.llms import LangchainLLMWrapper
    from langchain_cohere import ChatCohere
    evaluator_llm = LangchainLLMWrapper(ChatCohere(model="command-r-08-2024"))
    from langchain_cohere import CohereEmbeddings
    embeddings = CohereEmbeddings(
        model="embed-multilingual-v3.0",
    )

    contexts = extract_contexts(citations, docs_reranked)
  



    input = SingleTurnSample(
          user_input=question,
          response=answer,
          retrieved_contexts= contexts
      )
    scorer = FaithfulnesswithHHEM()
    await scorer.single_turn_ascore(input)
  
