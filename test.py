from transformers import GPT2Tokenizer, GPT2LMHeadModel
import logging
from utils import logging_config

logging_config('log_model', 'generator_test.log')

class GPT2Generator:
    def __init__(self, model_path='gpt2'):
        self.model_path = model_path
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_answer(self, question, max_length=64):
        try:
            inputs = self.tokenizer.encode(question + self.tokenizer.eos_token, return_tensors='pt')
            outputs = self.model.generate(inputs, max_length=max_length, pad_token_id=self.tokenizer.eos_token_id)
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if answer.startswith(question):
                answer = answer[len(question):].strip()

            return answer
        except Exception as e:
            logging.error(f"Error generating answer: {e}")
            return "Sorry, I couldn't generate an answer."

def main():
    generator = GPT2Generator(model_path='fine_tuned_gpt2_model')
    
    while True:
        question = input("Masukkan pertanyaan (atau ketik 'exit' untuk keluar): ")
        
        if question.lower() == 'exit':
            print("Terminating the program...")
            break
        
        answer = generator.generate_answer(question)
        print(f"Jawaban: {answer}")

        # Log the result
        logging.info(f"Model: {generator.model_path}")
        logging.info(f"Pertanyaan: {question}")
        logging.info(f"Jawaban: {answer}")
        logging.info("------------------------------------------\n")

if __name__ == "__main__":
    main()
