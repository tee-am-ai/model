import torch
from torch.utils.data import Dataset

class QADataset(Dataset):
    def __init__(self, inputs, tokenizer, labels=None, max_length=64):
        self.inputs = inputs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if idx >= len(self.inputs):
            raise IndexError(f"Index {idx} out of range for dataset with length {len(self.inputs)}")
        encodings = self.tokenizer(self.inputs[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        
        item = {'input_ids': input_ids, 'attention_mask': attention_mask}
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        else:
            item['labels'] = input_ids  # for perplexity calculation
        
        return item
