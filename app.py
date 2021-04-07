from flask import Flask, render_template
from transformers import T5Tokenizer, AutoModelForCausalLM
from transformers import pipeline
import re
import json

app = Flask(__name__)

tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")
classifier = pipeline('text-generation', model=model, tokenizer=tokenizer)

@app.route("/")
def index():
    name = "Hoge"
    return render_template('index.html', title='Izumi the Rubber Duck')

@app.route("/rinna/<seed>")
def rinna(seed=None):
    result = classifier(seed,max_length=40)
    resText = result[0]['generated_text']
    resText = resText.replace(seed,'')
    m = re.search('^.+[.。?!？！]',resText)
    if m:
        resText = m.group()
    resobj = {'generatedText' : resText}
    return json.dumps(resobj)