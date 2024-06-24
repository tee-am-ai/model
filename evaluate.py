import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch
import csv
import logging
import os

log_dir = './logs/'
os.makedirs(log_dir, exist_ok=True) 

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    filename='./logs/training.log'
)

# Load the dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path, delimiter='|', names=['question', 'answer'], encoding='utf-8', quoting=csv.QUOTE_NONE)
    if 'question' not in df.columns or 'answer' not in df.columns:
        raise ValueError("Dataset CSV harus memiliki kolom 'question' dan 'answer'.")
    return df

# Prepare the dataset
class QADataset(torch.utils.data.Dataset):
    def __init__(self, inputs, tokenizer, max_length=64):
        self.inputs = inputs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        try:
            encodings = self.tokenizer(input_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
            input_ids = encodings['input_ids'].squeeze()
            attention_mask = encodings['attention_mask'].squeeze()
        except Exception as e:
            logging.error(f"Error processing input_text ({idx}): {input_text}")
            logging.error(f"Exception: {str(e)}")
            raise e
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': input_ids}

# Main training function
def train_gpt2(dataset_path, output_dir='./results', logging_dir='./logs', num_train_epochs=3, per_device_train_batch_size=2, warmup_steps=500, weight_decay=0.01):
    # Load dataset
    df = load_dataset(dataset_path)
    
    # Prepare dataset
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    inputs = df['question'] + tokenizer.eos_token + df['answer']
    dataset = QADataset(inputs, tokenizer)
    
    # Initialize model
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir=logging_dir,
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    # Train the model
    logging.info(f"Training started with {len(dataset)} examples.")
    trainer.train()
    logging.info("Training completed.")
    
    # Evaluate the model on validation dataset
    eval_inputs = df.sample(frac=0.2, random_state=42)  # 20% for evaluation
    eval_inputs.reset_index(drop=True, inplace=True)
    eval_dataset = QADataset(eval_inputs['question'] + tokenizer.eos_token + eval_inputs['answer'], tokenizer)
    eval_trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
    )
    eval_results = eval_trainer.evaluate()
    logging.info(f"Evaluation results: {eval_results}")
    
    # Save model and tokenizer
    model.save_pretrained("gpt2_model")
    tokenizer.save_pretrained("gpt2_model")
    logging.info(f"Model and tokenizer saved to gpt2_model")
    
    # Additional evaluation metrics (example)
    # Example: Calculate perplexity
    perplexity = torch.exp(torch.tensor(eval_results['eval_loss']))
    logging.info(f"Perplexity: {perplexity.item()}")
    
    # Example: Perform qualitative evaluation (manual inspection)
    # Perform qualitative evaluation on a few samples
    logging.info("Performing qualitative evaluation on a few samples:")
    for i in range(5):
        input_text = eval_inputs.iloc[i]['question'] + tokenizer.eos_token + eval_inputs.iloc[i]['answer']
        input_ids = tokenizer(input_text, truncation=True, padding='max_length', max_length=64, return_tensors="pt")['input_ids']
        generated_text = model.generate(input_ids, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, max_new_tokens=50)
        logging.info(f"Input: {input_text}")
        logging.info(f"Generated: {tokenizer.decode(generated_text[0], skip_special_tokens=True)}")

    logging.info("Evaluation completed.")

    
# Example usage:
if __name__ == "__main__":
    dataset_path = 'datasets/1-6.csv'
    train_gpt2(dataset_path)
    logging.info("-------------------\n")
