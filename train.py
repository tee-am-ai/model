import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from utils import QADataset
import csv

# Load the dataset
df = pd.read_csv('datasets/coba.csv', delimiter='|', names=['question', 'answer'], encoding='utf-8', quoting=csv.QUOTE_NONE, on_bad_lines='skip')

# Prepare the dataset
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Combine question and answer into a single string for training
inputs = df['question'] + tokenizer.eos_token + df['answer']

dataset = QADataset(inputs, tokenizer)

# Load model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results_coba',         
    num_train_epochs=3,              
    per_device_train_batch_size=2,  
    warmup_steps=200,                
    weight_decay=0.01,               
    logging_dir='./logs',            
)

# Create Trainer
trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=dataset,         
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('gpt2_model_coba')
tokenizer.save_pretrained('gpt2_model_coba')
