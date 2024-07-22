import pandas as pd
import csv
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from utils import QADataset

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
    output_dir='./result/fine_tuning_results2',  # Directory to save the model and checkpoints
    num_train_epochs=10,  # Number of training epochs
    per_device_train_batch_size=8,  # Batch size per device (GPU/CPU)
    warmup_steps=500,  # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # Weight decay for regularization
    logging_dir='./fine_tuning_logs2',  # Directory to save the logs
    logging_steps=10,  # Number of steps between logging
    save_steps=200,  # Number of steps between saving checkpoints
    save_total_limit=2,  # Maximum number of checkpoints to keep
    evaluation_strategy='steps',  # Strategy for evaluation (e.g., 'steps' or 'epoch')
    eval_steps=200,  # Number of steps between evaluations (if evaluation_strategy='steps')
    load_best_model_at_end=True,  # Load the best model when finished training
    metric_for_best_model='eval_loss',  # Metric to use for selecting the best model
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
