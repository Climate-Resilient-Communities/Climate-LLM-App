
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os,torch
import bitsandbytes as bnb
from datasets import load_dataset
from trl import SFTTrainer
from datasets import Dataset
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
import re
import wandb
from huggingface_hub import login


def get_translation_prompt(input_language, output_language, text):
  """Return a prompt for translation"""
  prompt = f"Translate from {input_language} to {output_language}: '{text}'"
  return prompt


def get_message_format(prompts):
  """Return a message format for the Aya model"""
  messages = []

  for p in prompts:
    messages.append(
        [{"role": "user", "content": p}]
      )

  return messages

def generate_aya_23(
      prompts,
      model,
      tokenizer,
      temperature=0.3,
      top_p=0.75,
      top_k=0,
      max_new_tokens=1024
    ):
  """Return a generated text from the Aya model"""

  messages = get_message_format(prompts)

  input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        padding=True,
        return_tensors="pt",
      )
  input_ids = input_ids.to(model.device)
  prompt_padded_len = len(input_ids[0])

  gen_tokens = model.generate(
        input_ids,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        do_sample=True,
      )

  # get only generated tokens
  gen_tokens = [
      gt[prompt_padded_len:] for gt in gen_tokens
    ]

  gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
  return gen_text

def aya_translation(input_language, output_language, text, model, tokenizer):
  """Return a translated text from the Aya model"""
  prompt = get_translation_prompt(input_language, output_language, text)
  prompts = [prompt]
  generations = generate_aya_23(prompts, model, tokenizer)
  return generations[0]

if __name__ == "__main__":
    # Make sure you have access to model on HuggingFace first 
    MODEL_NAME = "CohereForAI/aya-23-8b"
    login(token="YOUR_HUGGINGFACE_TOKEN")
    # Initialize Aya Model
    quantization_config = None
    attn_implementation = None

    aya_model = AutoModelForCausalLM.from_pretrained(
              MODEL_NAME,
              quantization_config=quantization_config,
              attn_implementation=attn_implementation,
              torch_dtype=torch.bfloat16,
              device_map="auto",
            )
    # Load tokenizer
    aya_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)   

    input_language = sys.argv[1]
    output_language = sys.argv[2]
    text = sys.argv[3]
    aya_translation(input_language, output_language, text, aya_model, aya_tokenizer)
