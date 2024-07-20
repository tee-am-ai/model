import pandas as pd  # Mengimpor library pandas untuk manipulasi data dalam format DataFrame
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
# Mengimpor berbagai kelas dari library transformers untuk:
# - Tokenisasi (GPT2Tokenizer)
# - Memuat model GPT-2 (GPT2LMHeadModel)
# - Pelatihan model (Trainer dan TrainingArguments)
# - Kolasi data untuk pelatihan (DataCollatorForLanguageModeling)
import csv  # Mengimpor modul csv untuk membaca dan menulis file CSV
from utils import QADataset  # Mengimpor kelas QADataset dari modul utils untuk dataset tanya jawab khusus

# Kode ini mengimpor library dan modul yang dibutuhkan untuk manipulasi data, 
# memuat dan melatih model GPT-2, memproses file CSV, serta menangani dataset tanya jawab khusus.


# Load the dataset
def filter_valid_rows(row):
    return len(row) == 2

name = 'clean'
with open(f'datasets/{name}.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='|')
    filtered_rows = [row for row in reader if filter_valid_rows(row)]

df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])

# Prepare the dataset
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Combine question and answer into a single string for training
inputs = df['question'] + tokenizer.eos_token + df['answer']

dataset = QADataset(inputs, tokenizer, max_length=64)

# Load model
model = GPT2LMHeadModel.from_pretrained(model_name)

# Define data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Define training arguments
training_args = TrainingArguments(
    output_dir=f'./result/results_{name}',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the model
model_path = f'model/gpt2_model_{name}'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)