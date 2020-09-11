import flask
from flask_cors import CORS, cross_origin
import os
from Retriever import Retriever

app = flask.Flask(__name__)
app.config["DEBUG"] = True
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# DATASET = 'humor_challenge_data/gpt2_tokens_test.txt'
# DATASET = 'humor_challenge_data/bot_data/aggregated.csv'
# DATASET = 'humor_challenge_data/bot_data/qa_pair_data.csv'
# DATASET = 'humor_challenge_data/bot_data/rjokescharacterlimit.csv'
DATASET = 'humor_challenge_data/bot_data/qa_total.csv'
TOKENIZED_DATASET = 'humor_challenge_data/bot_data/qa_total_tokenized.csv'
# DATASET = 'humor_challenge_data/bot_data/non_qa_total.csv'
# TOKENIZED_DATASET = 'humor_challenge_data/bot_data/non_qa_total_tokenized.csv'

retriever = Retriever(DATASET, TOKENIZED_DATASET) 


def log(metadata):
    with open('usagelog','a') as logfile:
        logfile.write(str(metadata['joketuple']) + "," +  str(metadata['feedback']) + "\n")

with app.app_context():
    @app.route('/')
    @cross_origin()
    def hello():
        return "testy mctest"
    
    @app.route('/feedback', methods=['POST'])
    @cross_origin()
    def feedback():
        if flask.request.method == "POST":
            metadata = flask.request.json
            log(metadata)
        return "saved"
        
    @app.route('/retrieve', methods=['POST'])
    @cross_origin()
    def retreive():
        if flask.request.method == 'POST':
            metadata = flask.request.json
            if isinstance(metadata["history"], str) :
                metadata["history"] = (metadata["history"])
            response = retriever.predict(metadata["history"])
            return response

if __name__ == "__main__":
    #init(DEFAULT_MODEL_PATH, DEVICE_JSON)
    #print("ready")
    #app.run(host='192.168.1.10')
    #app.run(host='192.168.1.10', port=4444)
    app.run(port=5000)
    #app.run(host='192.168.1.10', port=4444, ssl_context=('/home/tobias/fullchain1.pem', '/home/tobias/privkey1.pem'))

