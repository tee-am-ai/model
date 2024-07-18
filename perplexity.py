import pandas as pd  # Mengimpor library pandas untuk manipulasi data
import torch  # Mengimpor library PyTorch untuk keperluan deep learning
import csv  # Mengimpor modul csv untuk memproses file CSV
import logging  # Mengimpor modul logging untuk mencatat log

from transformers import GPT2Tokenizer, GPT2LMHeadModel  # Mengimpor GPT2Tokenizer dan GPT2LMHeadModel dari library transformers
from torch.utils.data import DataLoader  # Mengimpor DataLoader dari PyTorch untuk memproses batch data
from utils import QADataset, logging_config  # Mengimpor QADataset dan logging_config dari modul utils


logging_config('log_model', 'generator_perplexity.log')
    
model_path = 'model/fine_tuned_gpt2_model2'

# Load the dataset
def filter_valid_rows(row):
    return len(row) == 2

name = 'clean'
with open(f'datasets/{name}.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='|')
    filtered_rows = [row for row in reader if filter_valid_rows(row)]

df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])

# Prepare the dataset
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# Combine question and answer into a single string for evaluation
inputs = df['question'] + tokenizer.eos_token + df['answer']

dataset = QADataset(inputs, tokenizer)

# Load model
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()

# Calculate perplexity
def calculate_perplexity(model, dataset, batch_size=6):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=batch_size)
    total_loss = 0.0
    for i, batch in enumerate(data_loader):
        print(f"Processing batch {i}/{len(data_loader)}")
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()

perplexity = calculate_perplexity(model, dataset)
print(f'Perplexity: {perplexity}')

logging.info(f"model: {model_path}")
logging.info(f'Perplexity: {perplexity}')
logging.info("------------------------------------------\n")
