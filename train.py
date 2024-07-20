import pandas as pd  # Mengimpor library pandas untuk manipulasi data dalam format DataFrame
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
# Mengimpor berbagai kelas dari library transformers
import csv  # Mengimpor modul csv untuk membaca dan menulis file CSV
from utils import QADataset  # Mengimpor kelas QADataset dari modul utils untuk dataset tanya jawab khusus

# Fungsi untuk memfilter baris yang valid dari dataset
def filter_valid_rows(row):
    return len(row) == 2

# Memuat dataset
name = 'clean'
with open(f'datasets/{name}.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='|')
    filtered_rows = [row for row in reader if filter_valid_rows(row)]  # Memfilter baris yang valid

df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])  # Membuat DataFrame dari baris yang difilter

# Mempersiapkan dataset
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)  # Memuat tokenizer GPT-2
tokenizer.pad_token = tokenizer.eos_token  # Menetapkan token padding sebagai token akhir (eos)

# Menggabungkan pertanyaan dan jawaban menjadi satu string untuk pelatihan
inputs = df['question'] + tokenizer.eos_token + df['answer']  # Menggabungkan pertanyaan dan jawaban dengan token akhir

# Membuat dataset untuk QA
dataset = QADataset(inputs, tokenizer, max_length=64)  # Menginisialisasi dataset dengan input dan tokenizer

# Memuat model
model = GPT2LMHeadModel.from_pretrained(model_name)  # Memuat model GPT-2

# Mendefinisikan data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Tidak menggunakan Masked Language Modeling
)

# Mendefinisikan argumen pelatihan
training_args = TrainingArguments(
    output_dir=f'./result/results_{name}',  # Direktori untuk menyimpan hasil
    num_train_epochs=3,  # Jumlah epoch pelatihan
    per_device_train_batch_size=4,  # Ukuran batch per perangkat
    warmup_steps=500,  # Jumlah langkah warmup
    weight_decay=0.01,  # Nilai weight decay
    logging_dir='./logs',  # Direktori untuk menyimpan log
    logging_steps=10,  # Interval langkah untuk logging
    save_steps=500,  # Interval langkah untuk menyimpan model
    save_total_limit=2,  # Batas total jumlah model yang disimpan
)

# Membuat Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# Melatih model
trainer.train()

# Menyimpan model
model_path = f'model/gpt2_model_{name}'
model.save_pretrained(model_path)  # Menyimpan model yang telah dilatih
tokenizer.save_pretrained(model_path)  # Menyimpan tokenizer
