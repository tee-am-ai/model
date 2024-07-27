from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Muat model dan tokenizer yang telah dilatih
model_path = "path/to/save/model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)