import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch
from utils import QADataset

# Load the dataset
df = pd.read_csv('datasets/1-6.csv', delimiter='|', names=['question', 'answer'])

# Prepare the dataset
tokenizer = GPT2Tokenizer.from_pretrained('fine_tuned_gpt2_model')
tokenizer.pad_token = tokenizer.eos_token

# Combine question and answer into a single string for training
inputs = df['question'] + tokenizer.eos_token + df['answer']

dataset = QADataset(inputs, tokenizer)

# Load model
model = GPT2LMHeadModel.from_pretrained('fine_tuned_gpt2_model')

# Define training arguments for fine-tuning
fine_tuning_args = TrainingArguments(
    output_dir='./fine_tuning_results2',
    num_train_epochs=5,                  # Meningkatkan jumlah epoch
    per_device_train_batch_size=8,       # Meningkatkan batch size
    warmup_steps=500,                    # Meningkatkan warmup steps
    weight_decay=0.01,                   # Menyesuaikan weight decay
    logging_dir='./fine_tuning_logs2',
    learning_rate=5e-5,                  # Menyesuaikan learning rate
)

# Create Trainer
trainer = Trainer(
    model=model,                         
    args=fine_tuning_args,                  
    train_dataset=dataset,         
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('fine_tuned_gpt2_model2')
tokenizer.save_pretrained('fine_tuned_gpt2_model2')
