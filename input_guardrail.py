
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets
from tqdm.auto import tqdm
from datasets import Dataset
import ray
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def construct_dataset(question):
  """Return a dataset from a question"""
  ds = Dataset.from_dict({'question': question})
  return ds

@ray.remote
def topic_moderation(question, pipe):
  """Return a topic moderation label from a question"""
  ds = construct_dataset(question)
  outs = pipe(KeyDataset(ds, "question"), padding=True, truncation=True)
  for out in outs:
    print(out)
    if out['label'] == 'no' and out['score'] >= 0.5:
      return "no"
    else:
      return "yes"

def get_class_probabilities(model, tokenizer, text, temperature=1.0, device='cpu'):
    """
    Evaluate the model on the given text with temperature-adjusted softmax.
    Note, as this is a DeBERTa model, the input text should have a maximum length of 512.
    
    Args:
        text (str): The input text to classify.
        temperature (float): The temperature for the softmax function. Default is 1.0.
        device (str): The device to evaluate the model on.
        
    Returns:
        torch.Tensor: The probability of each class adjusted by the temperature.
    """
    # Encode the text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    # Get logits from the model
    with torch.no_grad():
        logits = model(**inputs).logits
    # Apply temperature scaling
    scaled_logits = logits / temperature
    # Apply softmax to get probabilities
    probabilities = softmax(scaled_logits, dim=-1)

    predicted_class_id = logits.argmax().item()
    out = model.config.id2label[predicted_class_id]

    return {'out': out, 'probability': probabilities}

def get_jailbreak_score(model, tokenizer, text, temperature=1.0, device='cpu'):
    """
    Evaluate the probability that a given string contains malicious jailbreak or prompt injection.
    Appropriate for filtering dialogue between a user and an LLM.
    
    Args:
        text (str): The input text to evaluate.
        temperature (float): The temperature for the softmax function. Default is 1.0.
        device (str): The device to evaluate the model on.
        
    Returns:
        float: The probability of the text containing malicious content.
    """
    out_dict = get_class_probabilities(model, tokenizer, text, temperature, device)
    out = out_dict['out']
    probabilities = out_dict['probability']

    return out, probabilities[0, 2].item()

def get_indirect_injection_score(model, tokenizer, text, temperature=1.0, device='cpu'):
    """
    Evaluate the probability that a given string contains any embedded instructions (malicious or benign).
    Appropriate for filtering third party inputs (e.g., web searches, tool outputs) into an LLM.
    
    Args:
        text (str): The input text to evaluate.
        temperature (float): The temperature for the softmax function. Default is 1.0.
        device (str): The device to evaluate the model on.
        
    Returns:
        float: The combined probability of the text containing malicious or embedded instructions.
    """
    out_dict = get_class_probabilities(model, tokenizer, text, temperature, device)
    out = out_dict['out']
    probabilities = out_dict['probability']

    return out, (probabilities[0, 1] + probabilities[0, 2]).item()

@ray.remote
def prompt_guard(model, tokenizer, text):
    """Evaluate whether the question is safe to answer"""
    jailbreak_out, jailbreak_prob = get_jailbreak_score(model, tokenizer, text=text)
    injection_out, injection_prob = get_indirect_injection_score(model, tokenizer, text=text)
    if (jailbreak_out == 'JAILBREAK' and jailbreak_prob >= 0.9) or (injection_out == 'INJECTION' and injection_prob >= 0.9):
      return "yes"
    else:
      return "no"

if __name__=="__main__":
    question = sys.argv[1]
    
    # Load model and tokenizer for ClimateBERT
    climatebert_model_name = "climatebert/distilroberta-base-climate-detector"
    climatebert_model = AutoModelForSequenceClassification.from_pretrained(climatebert_model_name)
    climatebert_tokenizer = AutoTokenizer.from_pretrained(climatebert_model_name, max_len=512)

    # Load model and tokenizer for Prompt Guard
    promptguard_model_id = "meta-llama/Prompt-Guard-86M"
    promptguard_tokenizer = AutoTokenizer.from_pretrained(promptguard_model_id)
    promptguard_model = AutoModelForSequenceClassification.from_pretrained(promptguard_model_id)

    # Set up topic moderation pipeline
    topic_moderation_pipe = pipeline("text-classification", model=climatebert_model, tokenizer=climatebert_tokenizer)
    
    # Run topic moderation and prompt guards
    ray.init()
    topic_moderation_ret_id = topic_moderation.remote(question, topic_moderation_pipe)
    prompt_guard_ret_id = prompt_guard.remote(promptguard_model, promptguard_tokenizer, question)

    # Get results 
    topic_moderation_ret, topic_moderation_ret = ray.get([topic_moderation_ret_id, prompt_guard_ret_id])
    print(topic_moderation_ret, topic_moderation_ret)
