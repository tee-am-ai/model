from transformers import GPT2Tokenizer, GPT2LMHeadModel
import logging
import os

# Buat folder logs jika belum ada
if not os.path.exists('log_model'):
    os.makedirs('log_model')

# Konfigurasi logging
logging.basicConfig(
    filename='log_model/generator.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class GPT2Generator:
    def __init__(self, model_path='gpt2'):
        self.model_path = model_path
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    def generate_answer(self, question, max_length=100):
        # Encode the input question
        inputs = self.tokenizer.encode(question + self.tokenizer.eos_token, return_tensors='pt')
        # Generate the answer using the model
        outputs = self.model.generate(inputs, max_length=max_length, pad_token_id=self.tokenizer.eos_token_id)
        # Decode the generated answer
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if answer.startswith(question):
            answer = answer[len(question):].strip()

        return answer

# Example usage:
def main():
    generator = GPT2Generator(model_path='fine_tuned_gpt2_model')
    question = "aku penat dengan kehidupan ini"
    answer = generator.generate_answer(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")

    logging.info(f"model: {generator.model_path}")
    logging.info(f"Question: {question}")
    logging.info(f"Answer: {answer}")
    logging.info("-------------------\n")

if __name__ == "__main__":
    main()
