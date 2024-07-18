import os  # Import module for interacting with the operating system
import logging  # Import module for logging messages
from transformers import GPT2LMHeadModel, GPT2Tokenizer  # Import classes from the transformers library
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # Import functions for BLEU score calculation
from utils import logging_config  # Import custom logging configuration from utils module


logging_config('log_model', 'generator_bleu.log')

class GPT2Generator:
    def __init__(self, model_path='gpt2'):
        self.model_path = model_path
        self.model = GPT2LMHeadModel.from_pretrained(model_path)  # Inisialisasi model GPT-2 dari path yang diberikan
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)  # Inisialisasi tokenizer GPT-2 dari path yang diberikan

    def generate_answer(self, question, max_length=120):
        # Metode untuk menghasilkan jawaban berdasarkan pertanyaan yang diberikan
        # Encode pertanyaan input menggunakan tokenizer
        inputs = self.tokenizer.encode(question + self.tokenizer.eos_token, return_tensors='pt')
        # Generate jawaban menggunakan model GPT-2
        outputs = self.model.generate(inputs, max_length=max_length, pad_token_id=self.tokenizer.eos_token_id)
        # Decode jawaban yang dihasilkan
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Memastikan jawaban tidak dimulai dengan pertanyaan
        if answer.startswith(question):
            answer = answer[len(question):].strip()

        return answer
    
    def calculate_bleu_score(self, reference, generated):
        # Metode untuk menghitung skor BLEU antara referensi dan teks yang dihasilkan
        reference_tokens = self.tokenizer.tokenize(reference)
        generated_tokens = self.tokenizer.tokenize(generated)
        smoothing_function = SmoothingFunction().method4  # Menggunakan metode smoothing untuk skor BLEU
        bleu_score = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smoothing_function)
        return bleu_score


# Example usage:
def main():
    generator = GPT2Generator(model_path='gpt2_model_coba')

    # Example usage to generate and evaluate BLEU score
    example_question = "ibukota indonesia apa?"
    reference_answer = "jakarta"

    generated_answer = generator.generate_answer(example_question)
    print(f"Question: {example_question}")
    print(f"Answer: {generated_answer}")

    bleu_score = generator.calculate_bleu_score(reference_answer, generated_answer)
    print("BLEU Score:", bleu_score)

    logging.info(f"model: {generator.model_path}")
    logging.info(f"Question: {example_question}")
    logging.info(f"Answer: {generated_answer}")
    logging.info(f"Reference: {reference_answer}")
    logging.info(f"BLEU Score: {bleu_score}")
    logging.info("------------------------------------------\n")

if __name__ == "__main__":
    main()
