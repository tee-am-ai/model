import pandas as pd # type: ignore
import torch # type: ignore
import csv
import logging
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader # type: ignore
from utils import QADataset, logging_config

def load_dataset(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter='|')
            filtered_rows = [row for row in reader if len(row) == 2]
        return pd.DataFrame(filtered_rows, columns=['question', 'answer'])
