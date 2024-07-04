import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader
import torch
from utils import QADataset

# Load the dataset
df = pd.read_csv('datasets/clean.csv', delimiter='|', names=['question', 'answer'])

# Prepare the dataset
tokenizer = GPT2Tokenizer.from_pretrained('fine_tuned_gpt2_model1')
tokenizer.pad_token = tokenizer.eos_token

# Combine question and answer into a single string for evaluation
inputs = df['question'] + tokenizer.eos_token + df['answer']

dataset = QADataset(inputs, tokenizer)

# Load model
model = GPT2LMHeadModel.from_pretrained('fine_tuned_gpt2_model1')
model.eval()

# Calculate perplexity
def calculate_perplexity(model, dataset, batch_size=2):
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
