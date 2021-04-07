from flask import Flask
from transformers import T5Tokenizer, AutoModelForCausalLM

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello Flask, on Azure App Service for Linux"

@app.route("/rinna")
def rinna():
    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
    model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")
    return "Hello Rinna"