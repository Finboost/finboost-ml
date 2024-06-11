import pandas as pd
from datasets import Dataset
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, DataCollatorForLanguageModeling, create_optimizer
from transformers import TrainingArguments
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import re

# Load the dataset
df = pd.read_csv('../data/generative-ai/finansial-dataset.v2.csv')
# df = pd.read_csv('https://raw.githubusercontent.com/Finboost/finboost-ml/main/data/finansial-dataset-v2.csv')
dataset = Dataset.from_pandas(df[['text']])

# Load the tokenizer and model for fine-tuning
model_name = "cahya/gpt2-small-indonesian-522M"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
model = TFGPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))  # Update the model's token embeddings

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)  # Reduce max_length to 256

tokenized_datasets = dataset.map(tokenize_function, batched=True)

train_test_split = tokenized_datasets.train_test_split(test_size=0.01)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Convert datasets to tf.data.Dataset
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

tf_train_dataset = train_dataset.to_tf_dataset(
    columns=['input_ids', 'attention_mask', 'labels'],
    shuffle=True,
    batch_size=4,
    collate_fn=data_collator
)

tf_eval_dataset = eval_dataset.to_tf_dataset(
    columns=['input_ids', 'attention_mask', 'labels'],
    shuffle=False,
    batch_size=4,
    collate_fn=data_collator
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./models/fine_tuned_model",
    overwrite_output_dir=True,
    num_train_epochs=15,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Create optimizer and compile model
num_train_steps = len(tf_train_dataset) * training_args.num_train_epochs
optimizer, lr_schedule = create_optimizer(
    init_lr=training_args.learning_rate,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
    weight_decay_rate=training_args.weight_decay,
)

model.compile(optimizer=optimizer)

# Print model summary
model.summary()

# Train the model
history = model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=training_args.num_train_epochs)

# Save the model at the end
model.save_pretrained("../models/gen-ai/fine_tuned_model")
tokenizer.save_pretrained("../models/gen-ai/fine_tuned_model")