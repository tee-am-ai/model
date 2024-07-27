import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import csv
from utils import QADataset

# Load the dataset
def filter_valid_rows(row):
    return len(row) == 2

name = 'clean'
with open(f'datasets/{name}.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='|')
    filtered_rows = [row for row in reader if filter_valid_rows(row)]

df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])
