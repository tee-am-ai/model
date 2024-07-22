import logging
from utils import logging_config, GPT2Generator

logging_config('log_model', 'generator_test.log')

def main():
    example_question = "ibukota indonesia apa?"
    reference_answer = "jakarta"

    generated_answer = generator.generate_answer(example_question, max_length=100)
    print(f"Question: {example_question}")
    print(f"Answer: {generated_answer}")

    bleu_score = generator.calculate_bleu_score(reference_answer, generated_answer)
    print("BLEU Score:", bleu_score)
    
        # Log the result
        logging.info(f"Model: {generator.model_path}")
        logging.info(f"Pertanyaan: {question}")
        logging.info(f"Jawaban: {answer}")
        logging.info("------------------------------------------\n")

if __name__ == "__main__":
    main()
