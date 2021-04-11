from transformers import T5Tokenizer, AutoModelForCausalLM

tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")

tokenizer.save_pretrained(r"C:\repos\Izumi\model")
model.save_pretrained(r"C:\repos\Izumi\model")
