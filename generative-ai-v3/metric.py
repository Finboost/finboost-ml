import os
import numpy as np
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import tensorflow as tf
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load model and tokenizer
model_name = "./models/gen-ai/"  # Ganti dengan path model Anda
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = TFGPT2LMHeadModel.from_pretrained(model_name)

# Function to normalize prompts by removing punctuation and converting to lowercase
def normalize_prompt(prompt):
    import string
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

# Example usage of the calculate_perplexity function
if __name__ == "__main__":
    # Contoh teks untuk menghitung perplexity
    sample_text = "Ini adalah contoh teks untuk menghitung perplexity."

    # Normalisasi teks
    normalized_text = normalize_prompt(sample_text)
    
    # Hitung perplexity
    perplexity = calculate_perplexity(model, tokenizer, normalized_text)
    
    print("Perplexity:", perplexity)
