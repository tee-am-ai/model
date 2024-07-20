import logging  # Mengimpor modul logging untuk mencatat log
from utils import logging_config, GPT2Generator  # Mengimpor fungsi logging_config dan kelas GPT2Generator dari modul utils

# Konfigurasi logging
logging.basicConfig(filename='generator_bleu.log', level=logging.INFO)

# Fungsi utama
def main():
    generator = GPT2Generator(model_path='gpt2_model_coba')  # Membuat instance generator GPT-2 dengan path model yang ditentukan

    # Contoh penggunaan untuk menghasilkan dan mengevaluasi skor BLEU
    example_question = "ibukota indonesia apa?"  # Pertanyaan contoh
    reference_answer = "jakarta"  # Jawaban referensi

    # Menghasilkan jawaban menggunakan generator GPT-2
    generated_answer = generator.generate_answer(example_question, max_length=100)
    print(f"Question: {example_question}")
    print(f"Answer: {generated_answer}")

    # Menghitung skor BLEU antara jawaban referensi dan jawaban yang dihasilkan
    bleu_score = generator.calculate_bleu_score(reference_answer, generated_answer)
    print("BLEU Score:", bleu_score)

    # Mencatat informasi ke dalam log
    logging.info(f"Model: {generator.model_path}")
    logging.info(f"Question: {example_question}")
    logging.info(f"Answer: {generated_answer}")
    logging.info(f"Reference: {reference_answer}")
    logging.info(f"BLEU Score: {bleu_score}")
    logging.info("------------------------------------------\n")

# Memastikan fungsi main() berjalan jika skrip dieksekusi secara langsung
if __name__ == "__main__":
    main()
