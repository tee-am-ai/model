from torch.utils.data import Dataset  # Mengimpor kelas Dataset dari PyTorch untuk membuat dataset kustom
from transformers import GPT2Tokenizer, GPT2LMHeadModel  # Mengimpor GPT2Tokenizer dan GPT2LMHeadModel dari library transformers untuk tokenisasi dan model GPT-2
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # Mengimpor fungsi untuk menghitung skor BLEU dari NLTK
import logging  # Mengimpor modul logging untuk mencatat log
import os  # Mengimpor modul os untuk operasi terkait sistem operasi

# Mendefinisikan kelas GPT2Generator
class GPT2Generator:
    def __init__(self, model_path='gpt2'):
        self.model_path = model_path  # Menyimpan path model
        self.model = GPT2LMHeadModel.from_pretrained(model_path)  # Memuat model GPT-2 dari path
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    def generate_answer(self, question, max_length):
        # Mengencode pertanyaan dengan tokenizer dan menambahkan token EOS
        inputs = self.tokenizer.encode(question + self.tokenizer.eos_token, return_tensors='pt')
        # Menghasilkan jawaban dengan model
        outputs = self.model.generate(inputs, max_length=max_length, pad_token_id=self.tokenizer.eos_token_id)
        # Mendecode hasil output dari model menjadi teks
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Menghapus bagian pertanyaan dari jawaban jika ada
        if answer.startswith(question):
            answer = answer[len(question):].strip()

        return answer
    
    def calculate_bleu_score(self, reference, generated):
        # Tokenisasi referensi dan jawaban yang dihasilkan
        reference_tokens = self.tokenizer.tokenize(reference)
        generated_tokens = self.tokenizer.tokenize(generated)
        # Menghitung skor BLEU dengan smoothing
        smoothing_function = SmoothingFunction().method4
        bleu_score = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smoothing_function)
        return bleu_score


# Mendefinisikan kelas QADataset
class QADataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts  # Daftar teks (pertanyaan + jawaban)
        self.tokenizer = tokenizer  # Tokenizer yang digunakan
        self.max_length = max_length  # Panjang maksimum token

    def __len__(self):
        return len(self.texts)  # Mengembalikan jumlah teks

    def __getitem__(self, idx):
        # Mengencode teks pada indeks tertentu dengan padding dan truncation
        encodings = self.tokenizer(self.texts[idx], truncation=True, padding=True, max_length=self.max_length, return_tensors='pt')
        input_ids = encodings.input_ids[0]  # ID input
        attention_mask = encodings.attention_mask[0]  # Masker perhatian
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids}


# Fungsi untuk mengonfigurasi logging
def logging_config(log_dir, log_filename):
    # Membuat direktori log jika belum ada
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Mengonfigurasi logging
    logging.basicConfig(
        filename=f'{log_dir}/{log_filename}',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
