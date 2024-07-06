import pandas as pd
import csv
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from utils import QADataset

# Load the dataset
def filter_valid_rows(row):
    return len(row) == 2

with open('datasets/clean.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='|')
    filtered_rows = [row for row in reader if filter_valid_rows(row)]

df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])

# Prepare the dataset
tokenizer = GPT2Tokenizer.from_pretrained('gpt2_model')
tokenizer.pad_token = tokenizer.eos_token

# Combine question and answer into a single string for training
inputs = df['question'] + tokenizer.eos_token + df['answer']

dataset = QADataset(inputs, tokenizer)

# Load model
model = GPT2LMHeadModel.from_pretrained('gpt2_model')

# Define data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
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
model.save_pretrained('fine_tuned_gpt2_model')
tokenizer.save_pretrained('fine_tuned_gpt2_model')
