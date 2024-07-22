from torch.utils.data import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import logging
import os

# Define the GPT2Generator class
class GPT2Generator:
    def __init__(self, model_path='gpt2'):
        self.model_path = model_path
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    def generate_answer(self, question, max_length):
        inputs = self.tokenizer.encode(question + self.tokenizer.eos_token, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=max_length, pad_token_id=self.tokenizer.eos_token_id)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if answer.startswith(question):
            answer = answer[len(question):].strip()

        return answer
    
    def calculate_bleu_score(self, reference, generated):
        reference_tokens = self.tokenizer.tokenize(reference)
        generated_tokens = self.tokenizer.tokenize(generated)
        smoothing_function = SmoothingFunction().method4
        bleu_score = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smoothing_function)
        return bleu_score


# Define the QADataset class
class QADataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encodings = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids}

# Logging configuration
def setup_logging(log_dir, log_filename):
    # Create directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure logging
    logging.basicConfig(
        filename=os.path.join(log_dir, log_filename),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )