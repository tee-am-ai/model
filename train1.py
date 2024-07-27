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
