import logging  # Mengimpor modul logging untuk mencatat log
from utils import logging_config, GPT2Generator  # Mengimpor fungsi logging_config dan kelas GPT2Generator dari modul utils

# Kode ini mengimpor modul logging dan dua komponen dari modul utils,
# yaitu fungsi logging_config untuk konfigurasi logging dan kelas GPT2Generator untuk menghasilkan teks menggunakan model GPT-2.

# Konfigurasi logging
logging.basicConfig(filename='generator_test.log', level=logging.INFO)

# Fungsi utama
def main():
    generator = GPT2Generator(model_path='model/gpt2_cobacoba-20-24')  # Membuat instance generator GPT-2 dengan path model yang ditentukan
    
    while True:  # Memulai loop untuk menerima input dari pengguna
        question = input("Masukkan pertanyaan (atau ketik 'exit' untuk keluar): ").strip()  # Meminta input pertanyaan dari pengguna
        
        if question.lower() == 'exit':  # Memeriksa apakah pengguna ingin keluar
            print("Terminating the program...")
            break  # Menghentikan loop dan keluar dari program
        
        answer = generator.generate_answer(question, max_length=100)  # Menghasilkan jawaban menggunakan generator GPT-2
        print(f"Jawaban: {answer}")  # Mencetak jawaban yang dihasilkan

        # Mencatat hasil ke dalam log
        logging.info(f"Model: {generator.model_path}")
        logging.info(f"Pertanyaan: {question}")
        logging.info(f"Jawaban: {answer}")
        logging.info("------------------------------------------\n")

# Memastikan fungsi main() berjalan jika skrip dieksekusi secara langsung
if __name__ == "__main__":
    main()
