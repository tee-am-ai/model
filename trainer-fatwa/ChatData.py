from torch.utils.data import Dataset
import json

class ChatData(Dataset):
    def __init__(self, path:str, tokenizer):
        self.data = json.load(open(path, "r"))
        
        self.X = []
        for i in self.data:
            for j in i['dialog']:
                self.X.append(j['text'])