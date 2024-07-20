import pandas as pd  # Mengimpor library pandas untuk manipulasi data dalam format DataFrame
import numpy as np  # Mengimpor library numpy untuk operasi numerik
import logging  # Mengimpor modul logging untuk mencatat log
import csv  # Mengimpor modul csv untuk membaca dan menulis file CSV
import torch  # Mengimpor library PyTorch untuk operasi tensor dan deep learning
import evaluate  # Mengimpor library evaluate (kemungkinan untuk evaluasi model NLP, seperti dari HuggingFace)
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
# Mengimpor berbagai kelas dari library transformers
from sklearn.model_selection import train_test_split as tts
# Mengimpor fungsi untuk membagi dataset menjadi set pelatihan dan pengujian
from utils import QADataset, logging_config
# Mengimpor kelas QADataset dari modul utils untuk dataset tanya jawab khusus dan fungsi logging_config untuk konfigurasi logging

# Konfigurasi logging
logging_config('log_model', 'generator_accuracy.log')

def filter_valid_rows(row):
    return len(row) == 2 and all(row)

# Memuat dataset
num = 'coba'
filtered_rows = []
with open(f'datasets/{num}.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='|', quoting=csv.QUOTE_NONE)
    for row in reader:
        if filter_valid_rows(row):
            filtered_rows.append(row)  # Menyimpan baris yang valid

df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])
train_df, test_df = tts(df, test_size=0.2, random_state=42)  # 80% untuk pelatihan, 20% untuk pengujian

# Reset index untuk memastikan pengindeksan yang kontinu
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Mempersiapkan dataset
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)  # Memuat tokenizer GPT-2
tokenizer.pad_token = tokenizer.eos_token  # Menetapkan token padding sebagai token akhir (eos)

# Menggabungkan pertanyaan dan jawaban menjadi satu string untuk pelatihan
inputs_train = train_df['question'] + tokenizer.eos_token + train_df['answer']
dataset_train = QADataset(inputs_train, tokenizer, max_length=64)  # Membuat dataset pelatihan

inputs_test = test_df['question'] + tokenizer.eos_token + test_df['answer']
dataset_test = QADataset(inputs_test, tokenizer, max_length=64)  # Membuat dataset pengujian

# Memuat model
model = GPT2LMHeadModel.from_pretrained(model_name)  # Memuat model GPT-2

# Mendefinisikan data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Tidak menggunakan Masked Language Modeling
)

epoch = 20
batch_size = 24
# Mendefinisikan argumen pelatihan
training_args = TrainingArguments(
    output_dir=f'./result/results_coba{num}-{epoch}-{batch_size}',  # Direktori untuk menyimpan hasil
    num_train_epochs=epoch,  # Jumlah epoch pelatihan
    per_device_train_batch_size=batch_size,  # Ukuran batch per perangkat untuk pelatihan
    per_device_eval_batch_size=4,  # Ukuran batch per perangkat untuk evaluasi
    learning_rate=5e-5,  # Laju pembelajaran
    warmup_steps=500,  # Jumlah langkah warmup
    weight_decay=0.01,  # Nilai weight decay
    logging_dir='./logs',  # Direktori untuk menyimpan log
    logging_steps=10,  # Interval langkah untuk logging
    save_steps=500,  # Interval langkah untuk menyimpan model
    save_total_limit=2,  # Batas total jumlah model yang disimpan
    fp16=True,  # Menggunakan pelatihan presisi campuran (16-bit floating point)
)

# Memuat metrik BLEU
bleu_metric = evaluate.load("bleu", trust_remote_code=True)

# Fungsi untuk menghitung metrik evaluasi
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)
    
    predictions = torch.argmax(logits, dim=-1)
    
    # Mengubah tensor menjadi 1D
    predictions = predictions.view(-1)
    labels = labels.view(-1)
    
    # Menghapus indeks yang diabaikan (-100) dalam labels
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]

    bleu = bleu_metric.compute(predictions=predictions, references=labels)

    return {
        "bleu": bleu,
    }

# Membuat Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Melatih model
trainer.train()

# Menyimpan model
path = f'model/gpt2_coba{num}-{epoch}-{batch_size}'
model.save_pretrained(path)
tokenizer.save_pretrained(path)

# Mengevaluasi model
eval_results = trainer.evaluate()

# Mencetak hasil evaluasi, termasuk BLEU
print(f"Evaluation results: {eval_results}")
logging.info(f"Model: {path}")
logging.info(f"Evaluation results: {eval_results}")
logging.info("------------------------------------------\n")
