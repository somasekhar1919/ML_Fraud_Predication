from crypt import methods
from dataclasses import dataclass
from distutils.log import debug
from ipaddress import ip_address
from matplotlib.pyplot import get
import pandas as pd
from csv import reader
import numpy as np
import os 
import csv
from flask import Flask, redirect , render_template,request, url_for
from predict import predict


path = "/home/juniorcoder/Documents/ML project/upload_tasks/raw.csv"

def run():
    feed = predict(path)
    #modelName,scores = models.save_model(feed)
    #print("Max Accuracy"+str(modelName[scores.index(max(scores))])+" = "+str(max(scores)))

    #return "Max Accuracy : "+str(modelName[scores.index(max(scores))])+" = "+str(max(scores))


app = Flask(__name__)

app.config["UPLOAD_PATH"] = "/home/juniorcoder/Documents/ML project/upload_tasks"

@app.route("/", methods = ["GET","POST"])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        # for fraud input
        f =  request.files['csv_file']
        f.filename = "raw.csv"
        f.save(os.path.join(app.config["UPLOAD_PATH"],f.filename))
        feed = run()
        os.remove("/home/juniorcoder/Documents/ML project/upload_tasks/raw.csv")
        print(feed)

        #for fraud output
        results = []
        #usr = request.form.get('user_csv')
        feed = pd.read_csv('/home/juniorcoder/Documents/ML project/upload_tasks/done/final.csv')  
           
        """feed = feed.to_dict(orient='index')""" # have to try with pandas

        feed = feed.to_csv()
        feed = feed.split('\n')
        reader = csv.DictReader(feed)   # convert to dict
        for row in reader:              # using csv module 
            results.append(dict(row))
        
        fieldnames  = [key for key in results[0].keys()]
        
        return render_template('index.html', results = results , fieldnames  = fieldnames, len = len , list = list)



if __name__ == "__main__":
   app.run(debug=True)
















"""path = "InputFile.csv"
feed = pre.preprocess_data(path)
models.save_model(feed)
"""



