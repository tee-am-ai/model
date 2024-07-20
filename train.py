import pandas as pd  # Import the pandas library for data manipulation
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
# Import various classes from the transformers library for tokenization, model loading, training, and data collation
import csv  # Import the csv module to process CSV files
from utils import QADataset  # Import QADataset from the utils module for a custom question-answer dataset

# This code imports necessary libraries and modules to manipulate data, process CSV files,
# and load specific classes from the transformers library to fine-tune a GPT-2 model for a question-answering dataset.


# Fungsi untuk memfilter baris yang valid (hanya baris dengan 2 elemen yang diterima)
def filter_valid_rows(row):
    return len(row) == 2

name = 'clean'

# Membaca dataset dari file CSV
with open(f'datasets/{name}.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='|')
    filtered_rows = [row for row in reader if filter_valid_rows(row)]

# Membuat DataFrame dari baris-baris yang difilter
df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])

# Menggunakan tokenizer dari model GPT-2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Mengatur token padding sebagai token akhir (eos)

# Menggabungkan pertanyaan dan jawaban menjadi satu string untuk pelatihan
inputs = df['question'] + tokenizer.eos_token + df['answer']

# Membuat dataset khusus untuk QA menggunakan class QADataset
dataset = QADataset(inputs, tokenizer)

# Memuat model GPT-2
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Mendefinisikan data collator untuk model language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Tidak menggunakan masked language modeling
)

# Mendefinisikan argumen pelatihan
training_args = TrainingArguments(
    output_dir=f'./result/results_{name}',  # Direktori output untuk menyimpan hasil
    num_train_epochs=3,  # Jumlah epoch pelatihan
    per_device_train_batch_size=4,  # Ukuran batch pelatihan per perangkat
    warmup_steps=500,  # Jumlah langkah pemanasan
    weight_decay=0.01,  # Nilai weight decay
    logging_dir='./logs',  # Direktori untuk menyimpan log
    logging_steps=10,  # Interval langkah untuk logging
    save_steps=500,  # Interval langkah untuk menyimpan model
    save_total_limit=2,  # Jumlah maksimum checkpoint yang disimpan
)

# Membuat objek Trainer untuk melatih model
trainer = Trainer(
    model=model,  # Model yang akan dilatih
    args=training_args,  # Argumen pelatihan
    train_dataset=dataset,  # Dataset pelatihan
    data_collator=data_collator,  # Data collator
)

# Melatih model
trainer.train()

# Menyimpan model yang telah dilatih
model_path = f'model/gpt2_model_{name}'
model.save_pretrained(model_path)  # Menyimpan model
tokenizer.save_pretrained(model_path)  # Menyimpan tokenizer
