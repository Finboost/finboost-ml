import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import tensorflow as tf
import string
import re

# Load model and tokenizer
model_name = "./models/gen-ai/"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = TFGPT2LMHeadModel.from_pretrained(model_name)

# Function to normalize prompts by removing punctuation and converting to lowercase
def normalize_prompt(prompt):
    return ''.join(char for char in prompt if char not in string.punctuation).lower()

# Function to calculate perplexity
def calculate_perplexity(model, tokenizer, text):
    # Tokenize input text
    encodings = tokenizer(text, return_tensors='tf')

    # Get the logits from the model
    logits = model(encodings['input_ids'], attention_mask=encodings['attention_mask']).logits

    # Shift the input ids and logits to align them for the calculation
    shift_logits = logits[:, :-1, :]
    shift_labels = encodings['input_ids'][:, 1:]

    # Calculate the loss
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_fn(shift_labels, shift_logits)

    # Calculate perplexity
    perplexity = tf.exp(tf.reduce_mean(loss))

    return perplexity.numpy()

# Load the dataset and create a normalized prompt-response dictionary
df = pd.read_csv('./data/finansial-dataset-v2.csv')
prompt_response_dict = {normalize_prompt(prompt): response for prompt, response in zip(df['prompt'], df['response'])}

# Example usage of the calculate_perplexity function
if __name__ == "__main__":
    sample_text = "bagaimana cara mengelola keuangan yang baik"
    print("Perplexity:", calculate_perplexity(model, tokenizer, sample_text))

    # Calculate perplexity for each prompt in the dataset
    perplexities = {prompt: calculate_perplexity(model, tokenizer, prompt) for prompt in prompt_response_dict.keys()}
    for prompt, perp in perplexities.items():
        print(f"Prompt: {prompt}\nPerplexity: {perp}\n")
