from flask import Flask, render_template, send_from_directory
from transformers import T5Tokenizer, AutoModelForCausalLM
from transformers import pipeline
import re
import json
import os
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

app = Flask(__name__)

#tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
#model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")
modelpath = os.environ["modelPath"]
tokenizer = T5Tokenizer.from_pretrained(modelpath)
model = AutoModelForCausalLM.from_pretrained(modelpath)

classifier = pipeline('text-generation', model=model, tokenizer=tokenizer)

@app.route("/")
def index():
    name = "Hoge"
    return render_template('index.html', title='王様の耳はロバの耳ロバの耳 (雑談AI「ろばみみ」)')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', )

@app.route("/rinna/<seed>")
def rinna(seed=None):
    result = classifier(seed,max_length=40)
    resText = result[0]['generated_text']
    resText = resText.replace(seed,'')
    m = re.search('^.+[.。?!？！]',resText)
    if m:
        resText = m.group()

    client = authenticate_client()
    documents = [resText]
    #response = client.analyze_sentiment(documents=documents)[0]
    resobj = {}
    resobj['generatedText'] = resText
    #resobj['sentiment'] = response.sentiment
    resobj['sentiment'] = ""
    
    return json.dumps(resobj)

def authenticate_client():
    ta_credential = AzureKeyCredential(os.environ["TEXT_API_KEY"])
    text_analytics_client = TextAnalyticsClient(
            endpoint=os.environ["TEXT_API_END_POINT"], 
            credential=ta_credential)
    return text_analytics_client