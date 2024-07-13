import pandas as pd
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")