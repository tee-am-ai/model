import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import csv
from utils import QADataset

# Load the dataset
def filter_valid_rows(row):
    return len(row) == 2

name = 'clean'
file_path = f'datasets/{name}.csv'

try:
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='|')
        filtered_rows = [row for row in reader if filter_valid_rows(row)]

    df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])
    print(df.head())  # Display the first few rows of the DataFrame to verify

except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")

# Prepare the dataset
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Combine question and answer into a single string for training
inputs = df['question'] + tokenizer.eos_token + df['answer']

dataset = QADataset(inputs, tokenizer, max_length=64)

# Load model
model = GPT2LMHeadModel.from_pretrained(model_name)

# Define the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,  # Assuming `tokenizer` is previously defined
    mlm=False,  # Masked Language Modeling, set to `True` for MLM tasks (e.g., BERT)
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
