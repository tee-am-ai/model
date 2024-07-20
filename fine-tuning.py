import pandas as pd  # Mengimpor library pandas untuk manipulasi data
import csv  # Mengimpor modul csv untuk membaca dan menulis file CSV
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
# Mengimpor berbagai kelas dari library transformers untuk tokenisasi, pemuatan model GPT-2, pelatihan model, dan kolasi data
from utils import QADataset  # Mengimpor kelas QADataset dari modul utils untuk dataset tanya jawab khusus

# Kode ini mengimpor library dan modul yang dibutuhkan untuk memanipulasi data dengan pandas,
# membaca dan menulis file CSV, serta mempersiapkan pelatihan model GPT-2 menggunakan dataset tanya jawab khusus (QADataset).


# Load the dataset
def filter_valid_rows(row):
    return len(row) == 2

name = 'clean'
with open(f'datasets/{name}.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='|')
    filtered_rows = [row for row in reader if filter_valid_rows(row)]

df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])

# Prepare the dataset
model_name = 'model/fine_tuned_gpt2_model1'
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
    output_dir='./result/fine_tuning_results2',
    num_train_epochs=10,
    per_device_train_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./fine_tuning_logs2',
    logging_steps=10,
    save_steps=200,
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
model_path = 'model/fine_tuned_gpt2_model2'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)