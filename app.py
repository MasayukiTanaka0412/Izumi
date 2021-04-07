from flask import Flask
from transformers import T5Tokenizer, AutoModelForCausalLM
from transformers import pipeline
import re

app = Flask(__name__)

tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")
classifier = pipeline('text-generation', model=model, tokenizer=tokenizer)

@app.route("/")
def hello():
    return "Hello Flask, on Azure App Service for Linux"

@app.route("/rinna/<seed>")
def rinna(seed=None):
    result = classifier(seed,max_length=40)
    resText = result[0]['generated_text']
    m = re.search('^.+[.。$]',resText)
    if m:
        return m.group()
    return resText