import pandas as pd 
import torch 
import csv
import logging
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader 
from utils import QADataset, logging_config

def load_dataset(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter='|')
            filtered_rows = [row for row in reader if len(row) == 2]
        return pd.DataFrame(filtered_rows, columns=['question', 'answer'])
    except FileNotFoundError:
        logging.error(f"File {file_path} not found.")
        return pd.DataFrame(columns=['question', 'answer'])

def prepare_tokenizer(model_path):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def combine_questions_answers(df, tokenizer):
    return df['question'] + tokenizer.eos_token + df['answer']

def prepare_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    return model

def calculate_perplexity(model, dataset, batch_size=6):
    data_loader = DataLoader(dataset, batch_size=batch_size)
    total_loss = 0.0
    for batch in data_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            total_loss += outputs.loss.item()
    avg_loss = total_loss / len(data_loader)
    return torch.exp(torch.tensor(avg_loss)).item()

def main():
    logging_config('log_model', 'generator_perplexity.log')
    df = load_dataset('datasets/clean.csv')
    tokenizer = prepare_tokenizer('model/fine_tuned_gpt2_model2')
    inputs = combine_questions_answers(df, tokenizer)
    dataset = QADataset(inputs, tokenizer, max_length=64)
    model = prepare_model('model/fine_tuned_gpt2_model2')
    perplexity = calculate_perplexity(model, dataset)
    logging.info(f"Model: model/fine_tuned_gpt2_model2")
    logging.info(f'Perplexity: {perplexity}')
    logging.info("------------------------------------------\n")
    print(f'Perplexity: {perplexity}')

if __name__ == "__main__":
    main()