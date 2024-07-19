import logging
from utils import logging_config, GPT2Generator

logging_config('log_model', 'generator_test.log')

def main():
    generator = GPT2Generator(model_path='model/gpt2_cobacoba-20-24')
    
    while True:
        question = input("Masukkan pertanyaan (atau ketik 'exit' untuk keluar): ").strip()
        
        # if question.lower() == 'exit':
        #     print("Terminating the program...")
        #     break
        
        # answer = generator.generate_answer(question, max_length=100)
        # print(f"Jawaban: {answer}")

        # Log the result
        logging.info(f"Model: {generator.model_path}")
        logging.info(f"Pertanyaan: {question}")
        logging.info(f"Jawaban: {answer}")
        logging.info("------------------------------------------\n")

if __name__ == "__main__":
    main()
