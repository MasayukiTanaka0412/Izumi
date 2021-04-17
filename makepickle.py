import pickle
from transformers import T5Tokenizer, AutoModelForCausalLM
from transformers import pipeline


modelpath ="C:\\repos\\rinnamodel"
tokenizer = T5Tokenizer.from_pretrained(modelpath)
model = AutoModelForCausalLM.from_pretrained(modelpath)

classifier = pipeline('text-generation', model=model, tokenizer=tokenizer)
result = classifier("テストだよ")


with open("C:\\repos\\rinnamodel\\tokenizer.dmp", 'wb') as f:
    pickle.dump(tokenizer, f)

with open("C:\\repos\\rinnamodel\\model.dmp", 'wb') as f:
    pickle.dump(model, f)
