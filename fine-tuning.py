import pandas as pd  # Mengimpor library pandas untuk manipulasi data
import csv  # Mengimpor modul csv untuk membaca dan menulis file CSV
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
# Mengimpor berbagai kelas dari library transformers untuk tokenisasi, pemuatan model GPT-2, pelatihan model, dan kolasi data
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
model_name = 'model/fine_tuned_gpt2_model1'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  
inputs = df['question'] + tokenizer.eos_token + df['answer']  # Menggabungkan pertanyaan dan jawaban dengan token akhir

# Membuat dataset untuk QA
dataset = QADataset(inputs, tokenizer, max_length=64)  # Menginisialisasi dataset dengan input dan tokenizer

# Memuat model
model = GPT2LMHeadModel.from_pretrained(model_name)  # Memuat model GPT-2 yang telah disesuaikan

# Mendefinisikan data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Menetapkan mlm (masked language modeling) ke False
)

# Mendefinisikan argumen pelatihan
training_args = TrainingArguments(
    output_dir='./result/fine_tuning_results2',  # Direktori output untuk hasil pelatihan
    num_train_epochs=10,  # Jumlah epoch pelatihan
    per_device_train_batch_size=8,  # Ukuran batch untuk setiap perangkat
    warmup_steps=500,  # Jumlah langkah warmup
    weight_decay=0.01,  # Koefisien peluruhan bobot
    logging_dir='./fine_tuning_logs2',  # Direktori untuk log pelatihan
    logging_steps=10,  # Jumlah langkah antara setiap logging
    save_steps=200,  # Jumlah langkah antara setiap penyimpanan model
    save_total_limit=2,  # Jumlah maksimum model yang disimpan
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

# Menyimpan model yang telah dilatih
model_path = 'model/fine_tuned_gpt2_model2'
model.save_pretrained(model_path)  # Menyimpan model yang telah dilatih
tokenizer.save_pretrained(model_path)  # Menyimpan tokenizer yang telah disesuaikan
