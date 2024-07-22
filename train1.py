import pandas as pd
import numpy as np
import logging
import csv
import torch
import evaluate
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
from utils import QADataset, logging_config

logging_config('log_model', 'generator_accuracy.log')

# Function to filter valid rows
def filter_valid_rows(row):
    return len(row) == 2 and all(row)

# Load the dataset
num = 'coba'
filtered_rows = []
with open(f'datasets/{num}.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='|', quoting=csv.QUOTE_NONE)
    for row in reader:
        if filter_valid_rows(row):
            filtered_rows.append(row)

df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])

# Split dataset into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Reset index to ensure continuous indexing
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Prepare the dataset
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Combine question and answer into a single string for training
inputs_train = train_df['question'] + tokenizer.eos_token + train_df['answer']
dataset_train = QADataset(inputs_train, tokenizer)

inputs_test = test_df['question'] + tokenizer.eos_token + test_df['answer']
dataset_test = QADataset(inputs_test, tokenizer)

# Load model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Define data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

epoch = 20
batch_size = 24
# Define training arguments
training_args = TrainingArguments(
    output_dir=f'./result/results_coba{num}-{epoch}-{batch_size}',
    num_train_epochs=epoch,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    fp16=True, 
    # eval_strategy="epoch",
    # eval_steps=500,
)

# Load metrics
accuracy_metric = evaluate.load("accuracy", trust_remote_code=True)
bleu_metric = evaluate.load("bleu", trust_remote_code=True)
rouge_metric = evaluate.load("rouge", trust_remote_code=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)
    
    predictions = torch.argmax(logits, dim=-1)
    
    # Flatten tensors to 1D
    predictions = predictions.view(-1)
    labels = labels.view(-1)
    
   

    return {
        "accuracy": accuracy,
        "bleu": bleu,
        "rouge": rouge,
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the model
path = f'model/gpt2_coba{num}-{epoch}-{batch_size}'
model.save_pretrained(path)
tokenizer.save_pretrained(path)

# Evaluate model
eval_results = trainer.evaluate()

# Print evaluation results, including accuracy
print(f"Evaluation results: {eval_results}")
logging.info(f"model: {path}")
logging.info(f"Evaluation results: {eval_results}")
logging.info("------------------------------------------\n")
