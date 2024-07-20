import pandas as pd  # Mengimpor library pandas untuk manipulasi data
import torch  # Mengimpor library PyTorch untuk keperluan deep learning
import csv  # Mengimpor modul csv untuk memproses file CSV
import logging  # Mengimpor modul logging untuk mencatat log
from transformers import GPT2Tokenizer, GPT2LMHeadModel  # Mengimpor GPT2Tokenizer dan GPT2LMHeadModel dari library transformers
from torch.utils.data import DataLoader  # Mengimpor DataLoader dari PyTorch untuk memproses batch data
from utils import QADataset, logging_config  # Mengimpor QADataset dan logging_config dari modul utils

# Konfigurasi logging dengan nama file 'log_model' dan file log 'generator_perplexity.log'
logging_config('log_model', 'generator_perplexity.log')

model_path = 'model/fine_tuned_gpt2_model2'  # Path model yang telah dilatih

# Memuat dataset
def filter_valid_rows(row):
    return len(row) == 2  # Memfilter baris valid yang memiliki 2 kolom

name = 'clean'
with open(f'datasets/{name}.csv', 'r', encoding='utf-8') as file:  # Membuka file CSV
    reader = csv.reader(file, delimiter='|')
    filtered_rows = [row for row in reader if filter_valid_rows(row)]  # Memfilter baris valid

df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])  # Membuat DataFrame dari baris yang difilter

# Mempersiapkan dataset
tokenizer = GPT2Tokenizer.from_pretrained(model_path)  # Memuat tokenizer dari model yang dilatih
tokenizer.pad_token = tokenizer.eos_token  # Mengatur token padding menjadi token akhir

# Menggabungkan pertanyaan dan jawaban menjadi satu string untuk evaluasi
inputs = df['question'] + tokenizer.eos_token + df['answer']

dataset = QADataset(inputs, tokenizer)  # Membuat dataset menggunakan kelas QADataset

# Memuat model
model = GPT2LMHeadModel.from_pretrained(model_path)  # Memuat model yang telah dilatih
model.eval()  # Mengatur model ke mode evaluasi

# Menghitung perplexity
def calculate_perplexity(model, dataset, batch_size=6):
    model.eval()  # Mengatur model ke mode evaluasi
    data_loader = DataLoader(dataset, batch_size=batch_size)  # Membuat DataLoader untuk memuat data
    total_loss = 0.0
    for i, batch in enumerate(data_loader):  # Mengiterasi batch data
        print(f"Processing batch {i}/{len(data_loader)}")
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        with torch.no_grad():  # Tidak menghitung gradien
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)  # Memprediksi keluaran
            loss = outputs.loss  # Menghitung loss
            total_loss += loss.item()  # Menambahkan loss ke total_loss
    avg_loss = total_loss / len(data_loader)  # Menghitung rata-rata loss
    perplexity = torch.exp(torch.tensor(avg_loss))  # Menghitung perplexity
    return perplexity.item()

perplexity = calculate_perplexity(model, dataset)  # Menghitung perplexity
print(f'Perplexity: {perplexity}')

# Logging informasi
logging.info(f"model: {model_path}")
logging.info(f'Perplexity: {perplexity}')
logging.info("------------------------------------------\n")

# Kode ini bertujuan untuk menghitung perplexity dari model GPT-2 yang telah dilatih,
# menggunakan dataset tanya jawab, dan mencatat hasilnya ke dalam log file.
