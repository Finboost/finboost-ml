# load library
# pip install accelerate -U

import pandas as pd
from datasets import Dataset
# from transformers import RobertaTokenizer, RobertaForQuestionAnswering, Trainer, TrainingArguments, default_data_collator
from transformers import RobertaTokenizerFast, RobertaForQuestionAnswering, Trainer, TrainingArguments, default_data_collator

# Load the dataset
# df = pd.read_csv('../data/dataset.csv')
df = pd.read_csv('./data/generative-ai/final-dataset.csv')
dataset = Dataset.from_pandas(df)

# Load the tokenizer and model
model_name = "deepset/roberta-base-squad2"
# tokenizer = RobertaTokenizer.from_pretrained(model_name)
tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
model = RobertaForQuestionAnswering.from_pretrained(model_name)

# Tokenize dataset
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )
    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answer"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = examples["answer_start"][i]
        end_char = start_char + len(answer)

        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_idx = context_start
            while start_idx <= context_end and offset[start_idx][0] <= start_char:
                start_idx += 1
            start_positions.append(start_idx - 1)

            end_idx = context_start
            while end_idx <= context_end and offset[end_idx][1] < end_char:
                end_idx += 1
            end_positions.append(end_idx - 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./models/roberta/fine_tuned_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    data_collator=default_data_collator,
)

# Fine-tune the model
trainer.train()

# Save the model
trainer.save_model("models/roberta/fine_tuned_model")

