import os
import sys
import tiktoken

def read_markdown_in_chunks(file_object):
    """Returns a 1MB chunk of a markdown file"""
    while True:
        chunk = file_object.read(1000000)
        if not chunk:
            break
        yield chunk


def read_full_markdown(file_path: str) -> str:
    """Returns the content of markdown file in the file_path as a string."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")

    try:
        content = ""
        with open(file_path, 'r', encoding='utf-8') as file:
            for chunk in read_markdown_in_chunks(file):
                content += chunk
        return content
    except Exception as e:
        raise Exception(f"An error occurred while reading the file: {e}")

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
