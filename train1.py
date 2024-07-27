import pandas as pd
import numpy as np
import logging
import csv
import torch
import evaluate
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split as tts
from utils import QADataset, logging_config

logging_config('log_model', 'generator_accuracy.log')

# Function to filter valid rows
def filter_valid_rows(row):
    return len(row) == 2 and all(row)

# Load the dataset
num = 'coba'
filtered_rows = []
with open(f'datasets/{num}.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='|', quoting=csv.QUOTE_NONE)
    for row in reader:
        if filter_valid_rows(row):
            filtered_rows.append(row)

df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])

# Split dataset into training and test sets
train_df, test_df = tts(df, test_size=0.2, random_state=42)

# Reset index to ensure continuous indexing
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Prepare the dataset
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Combine question and answer into a single string for training
inputs_train = train_df['question'] + tokenizer.eos_token + train_df['answer']
dataset_train = QADataset(inputs_train, tokenizer, max_length=64)

inputs_test = test_df['question'] + tokenizer.eos_token + test_df['answer']
dataset_test = QADataset(inputs_test, tokenizer, max_length=64)

# Load model
model = GPT2LMHeadModel.from_pretrained(model_name)

# Define data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

