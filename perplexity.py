import pandas as pd  # Mengimpor library pandas untuk manipulasi data
import torch  # Mengimpor library PyTorch untuk operasi tensor dan deep learning
import csv  # Mengimpor modul csv untuk membaca dan menulis file CSV
import logging  # Mengimpor modul logging untuk mencatat log
from transformers import GPT2Tokenizer, GPT2LMHeadModel  # Mengimpor GPT2Tokenizer dan GPT2LMHeadModel dari library transformers
from torch.utils.data import DataLoader  # Mengimpor DataLoader dari PyTorch untuk memuat data dalam batch
from utils import QADataset, logging_config  # Mengimpor QADataset dan logging_config dari modul utils

# Konfigurasi logging
logging.basicConfig(filename='generator_perplexity.log', level=logging.INFO)

# Fungsi untuk memfilter baris yang valid dari dataset
def filter_valid_rows(row):
    return len(row) == 2  # Memastikan bahwa baris memiliki tepat 2 elemen

# Memuat dataset
name = 'clean'
with open(f'datasets/{name}.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='|')
    filtered_rows = [row for row in reader if filter_valid_rows(row)]  # Memfilter baris yang valid

df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])  # Membuat DataFrame dari baris yang difilter

# Mempersiapkan dataset
model_path = 'model/fine_tuned_gpt2_model2'
tokenizer = GPT2Tokenizer.from_pretrained(model_path)  # Memuat tokenizer dari model yang telah disesuaikan
tokenizer.pad_token = tokenizer.eos_token  # Menetapkan token padding sebagai token akhir (eos)

# Menggabungkan pertanyaan dan jawaban menjadi satu string untuk evaluasi
inputs = df['question'] + tokenizer.eos_token + df['answer']  # Menggabungkan pertanyaan dan jawaban dengan token akhir

# Membuat dataset untuk QA
dataset = QADataset(inputs, tokenizer, max_length=64)  # Menginisialisasi dataset dengan input dan tokenizer

# Memuat model
model = GPT2LMHeadModel.from_pretrained(model_path)  # Memuat model GPT-2 yang telah disesuaikan
model.eval()  # Menetapkan mode evaluasi untuk model

# Fungsi untuk menghitung perplexity
def calculate_perplexity(model, dataset, batch_size=6):
    model.eval()  # Menetapkan mode evaluasi untuk model
    data_loader = DataLoader(dataset, batch_size=batch_size)  # Membuat DataLoader dengan batch size yang ditentukan
    total_loss = 0.0
    for i, batch in enumerate(data_loader):  # Iterasi melalui DataLoader
        print(f"Processing batch {i}/{len(data_loader)}")
        input_ids = batch['input_ids']  # Mendapatkan input IDs dari batch
        attention_mask = batch['attention_mask']  # Mendapatkan attention mask dari batch
        with torch.no_grad():  # Tidak mengkalkulasi gradient
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)  # Mendapatkan output model
            loss = outputs.loss  # Mendapatkan loss dari output
            total_loss += loss.item()  # Menambahkan loss ke total loss
    avg_loss = total_loss / len(data_loader)  # Menghitung rata-rata loss
    perplexity = torch.exp(torch.tensor(avg_loss))  # Menghitung perplexity dari rata-rata loss
    return perplexity.item()

# Menghitung perplexity
perplexity = calculate_perplexity(model, dataset)
print(f'Perplexity: {perplexity}')

# Mencatat hasil perplexity ke dalam log
logging.info(f"Model: {model_path}")
logging.info(f'Perplexity: {perplexity}')
logging.info("------------------------------------------\n")
