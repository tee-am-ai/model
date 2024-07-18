import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from utils import logging_config

logging_config('log_model', 'generator_bleu.log')

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
