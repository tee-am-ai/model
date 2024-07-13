import pandas as pd
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def tokenize_text(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)

df = pd.read_csv("../dialogs.csv", delimiter='|')

df['text'] = df['question'] + " " + df['answer']

dataset = Dataset.from_pandas(df[['text']])

tokenized_dataset = dataset.map(tokenize_text, batched=True, remove_columns=["text"])

model = GPT2LMHeadModel.from_pretrained("gpt2")