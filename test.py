import logging  # Mengimpor modul logging untuk mencatat log
from utils import logging_config, GPT2Generator  # Mengimpor fungsi logging_config dan kelas GPT2Generator dari modul utils

# Mengkonfigurasi logging
logging.basicConfig(filename='generator_test.log', level=logging.INFO)

def main():
    generator = GPT2Generator(model_path='model/gpt2_cobacoba-20-24')
    
    while True:
        # Menerima input pertanyaan dari pengguna
        question = input("Masukkan pertanyaan (atau ketik 'exit' untuk keluar): ").strip()
        
        if question.lower() == 'exit':
            # Menghentikan program jika pengguna mengetik 'exit'
            print("Terminating the program...")
            break
        
        answer = generator.generate_answer(question, max_length=100)
        print(f"Jawaban: {answer}")

        # Mencatat hasil ke file log
        logging.info(f"Model: {generator.model_path}")
        logging.info(f"Pertanyaan: {question}")
        logging.info(f"Jawaban: {answer}")
        logging.info("------------------------------------------\n")

if __name__ == "__main__":
    main()
