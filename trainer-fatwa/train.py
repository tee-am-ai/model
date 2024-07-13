import pandas as pd
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token

def tokenize_text(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)

df = pd.read_csv("dialogs.csv", delimiter='|')

df['text'] = df['question'] + " " + df['answer']

dataset = Dataset.from_pandas(df[['text']])

tokenized_dataset = dataset.map(tokenize_text, batched=True, remove_columns=["text"])

model = GPT2LMHeadModel.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

trainer.save_model("trained_gpt2_model")
tokenizer.save_pretrained("trained_gpt2_model")