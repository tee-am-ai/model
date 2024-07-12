from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Muat model dan tokenizer yang telah dilatih
model_path = "path/to/save/model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Fungsi untuk menghasilkan teks
def generate_text(prompt, max_length=50):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Contoh pengujian
prompt = "Hari ini saya merasa sangat bahagia karena"
generated_text = generate_text(prompt)
print(generated_text)
