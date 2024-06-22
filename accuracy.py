import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch
import csv
from sklearn.model_selection import train_test_split
import numpy as np

# Muat dataset
df = pd.read_csv('dataset/0.csv', delimiter='|', names=['question', 'answer'], encoding='utf-8', quoting=csv.QUOTE_NONE)

# Siapkan dataset
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Gabungkan question dan answer menjadi satu string untuk pelatihan
inputs = df['question'] + tokenizer.eos_token + df['answer']

# Bagi dataset menjadi dataset pelatihan dan evaluasi
train_inputs, eval_inputs = train_test_split(inputs, test_size=0.25, random_state=42)

# Konversi ke pandas Series untuk menjaga indeks
train_inputs = pd.Series(train_inputs).reset_index(drop=True)
eval_inputs = pd.Series(eval_inputs).reset_index(drop=True)

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
            print(f"Error processing input_text ({idx}): {input_text}")
            raise e
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': input_ids}

train_dataset = QADataset(train_inputs, tokenizer)
eval_dataset = QADataset(eval_inputs, tokenizer)

# Muat model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Tentukan argumen pelatihan
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Buat Trainer
trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=eval_dataset,
)

# Latih model
trainer.train()

# Simpan model
model.save_pretrained('gpt2_model')
tokenizer.save_pretrained('gpt2_model')

# Fungsi untuk mengevaluasi akurasi
def evaluate_accuracy(model, tokenizer, eval_dataset):
    model.eval()
    correct = 0
    total = 0
    
    for idx in range(len(eval_dataset)):
        with torch.no_grad():
            input_data = eval_dataset[idx]
            input_ids = input_data['input_ids'].unsqueeze(0)
            attention_mask = input_data['attention_mask'].unsqueeze(0)
            labels = input_data['labels'].unsqueeze(0)

            outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=input_ids.shape[1] + 20)
            predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            target_text = tokenizer.decode(labels[0], skip_special_tokens=True)
            
            if predicted_text == target_text:
                correct += 1
            total += 1
    
    accuracy = correct / total
    return accuracy

# Evaluasi akurasi model
accuracy = evaluate_accuracy(model, tokenizer, eval_dataset)
print(f"Akurasi model: {accuracy * 100:.2f}%")
