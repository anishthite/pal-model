import flask
import torch
from flask_cors import CORS, cross_origin
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import generator
import os
from predictor import predict, init
import threading

DEFAULT_MODEL_PATH = "./models/medium"
DEVICE_JSON = {"device": "cpu"}
config = GPT2Config.from_json_file(os.path.join(DEFAULT_MODEL_PATH, 'config.json'))
model = GPT2LMHeadModel(config)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

app = flask.Flask(__name__)
app.config["DEBUG"] = True
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
init(DEFAULT_MODEL_PATH, DEVICE_JSON)
print('ready')

with app.app_context():
    
    @app.route('/')
    @cross_origin()
    def hello():
        return "hi anika"
    
    @app.route('/feedback', methods=['POST'])
    @cross_origin()
    def feedback():
        if flask.request.method == 'POST':
            data = flask.request.json
            formathread = threading.Thread(target=format_save, args=(data,))
            formathread.start()
        return 'saved'

    @app.route('/predict', methods=['POST'])
    @cross_origin()
    def prediction():
        if flask.request.method == 'POST':
            metadata = flask.request.json
            print(metadata)
            response = predict(metadata)
            savethread = threading.Thread(target = save, args=(metadata, response,))
            savethread.start()            
            return response

    def format_save(data):
        save_feedback(format_feedback(data))

    def format_feedback(data):
        data["history"] = data["history"][:-1]
        data["history"].append(data["feedback"])
        return data

    def save_feedback(data):
        with open('feedback.log', 'a') as log:
            log.write(''.join([str(data["id"]), str(data["history"]), '\n']))


    def save(metadata, response):
        with open('usage.log', "a") as log:
            metadata['history'].append(response)
            log.write(''.join([str(metadata["id"]), str(metadata["history"]), '\n']))



if __name__ == "__main__":
    #init(DEFAULT_MODEL_PATH, DEVICE_JSON)
    print("ready")
    #app.run(host='192.168.1.10')
    #app.run(host='192.168.1.10', port=4240)
    app.run(port=4240)
    #app.run(host='192.168.1.10', port=4240, ssl_context=('/home/tobias/fullchain1.pem', '/home/tobias/privkey1.pem'))
    
