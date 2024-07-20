import logging
from utils import logging_config, GPT2Generator

# Mengkonfigurasi logging dengan nama 'log_model' dan file log 'generator_test.log'
logging_config('log_model', 'generator_test.log')

def main():
    # Membuat instance dari GPT2Generator dengan path model yang telah dilatih
    generator = GPT2Generator(model_path='model/gpt2_cobacoba-20-24')
    
    while True:  # Memulai loop tak terbatas untuk menerima input dari pengguna
        question = input("Masukkan pertanyaan (atau ketik 'exit' untuk keluar): ")  # Menerima input pertanyaan dari pengguna
        
        if question.lower() == 'exit':  # Memeriksa apakah pengguna ingin keluar
            print("Terminating the program...")  # Mencetak pesan bahwa program akan berhenti
            break  # Keluar dari loop
        
        # Menghasilkan jawaban menggunakan metode generate_answer dari GPT2Generator
        answer = generator.generate_answer(question)
        print(f"Jawaban: {answer}")  # Mencetak jawaban ke layar

        # Mencatat hasil ke file log
        logging.info(f"Model: {generator.model_path}")  # Mencatat path model yang digunakan
        logging.info(f"Pertanyaan: {question}")  # Mencatat pertanyaan dari pengguna
        logging.info(f"Jawaban: {answer}")  # Mencatat jawaban yang dihasilkan
        logging.info("------------------------------------------\n")  # Mencatat garis pemisah untuk kejelasan log

if __name__ == "__main__":  # Memastikan bahwa fungsi main() hanya dijalankan jika skrip ini dieksekusi langsung
    main()  # Menjalankan fungsi utama
