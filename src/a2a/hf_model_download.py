from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "sshleifer/tiny-gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Model downloaded locally.")