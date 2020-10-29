import flask
from flask_cors import CORS, cross_origin
import os
#from retriever.Retriever import Retriever
from gpt2.gpt2run import HumorGenGPT 
from humorpipeline import HumorPipeline

app = flask.Flask(__name__)
app.config["DEBUG"] = True
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# DATASET = 'humor_challenge_data/gpt2_tokens_test.txt'
# DATASET = 'humor_challenge_data/bot_data/aggregated.csv'
# DATASET = 'humor_challenge_data/bot_data/qa_repair_data.csv'
# DATASET = 'humor_challenge_data/bot_data/rjokescharacterlimit.csv'
# DATASET = 'humor_challenge_data/bot_data/qa_total.csv'
#from Retriever import Retriever
DATASET = 'bert_train_data/qa_total.csv'
TOKENIZED_DATASET = 'humor_challenge_data/bot_data/qa_total_tokenized.txt'
# WORD2VEC_DATASET = 'humor_challenge_data/bot_data/qa_total_word2vec.csv'
# DATASET = 'humor_challenge_data/bot_data/non_qa_total.csv'
# TOKENIZED_DATASET = 'humor_challenge_data/bot_data/non_qa_total_tokenized.csv'
# 

#retriever = Retriever(DATASET, TOKENIZED_DATASET) 
generator = HumorGenGPT('/home/tobias/humor/pal-model/gpt2/trained_models/gpt2_tokens_tag_10086.pt')
dialogenerator = HumorGenGPT('/home/tobias/humor/pal-model/gpt2/trained_models/dialogpt2_tokens_tag.pt')

pipeline = HumorPipeline('/home/tobias/humor/pal-model/gpt2/trained_models/gpt2_tokens_tag_10086.pt', '/home/tobias/humor/pal-model/newsave.pt', num_gens=2)

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
 #           response = retriever.predict(metadata["history"])
            response = "please refresh the page"
            return response

    @app.route('/generate_gpt2_ind', methods=['POST'])
    @cross_origin()
    def generate_gpt2_ind():
        if flask.request.method == 'POST':
            metadata = flask.request.json
            if isinstance(metadata["history"], str):
                metadata["history"] = (metadata["history"])
            response = generator.predict(metadata["history"],top_k=30, do_sample=True)
            return response

    @app.route('/generate_dialogpt2', methods=['POST'])
    @cross_origin()
    def generate_dialogpt2():
        if flask.request.method == 'POST':
            metadata = flask.request.json
            if isinstance(metadata["history"], str):
                metadata["history"] = (metadata["history"])
            response = dialogenerator.predict(metadata["history"], top_k=30, do_sample=True)
            return response
    
    @app.route('/generate_pipeline', methods=['POST'])
    @cross_origin()
    def generate_pipeline():
        if flask.request.method == 'POST':
            metadata = flask.request.json
            if isinstance(metadata["history"], str):
                metadata["history"] = (metadata["history"])
            response = pipeline.predict(metadata["history"], do_sample=True)
            return response

    @app.route('/classify', methods=['POST'])
    @cross_origin()
    def classify():
        if flask.request.method == 'POST':
            metadata = flask.request.json
            if isinstance(metadata["history"], str):
                metadata["history"] = (metadata["history"])
            response = pipeline.classifer(metadata["history"])
            score = response[0].numpy()[1]  
            if score > 0.7:
                return 'I love it! It is ' + str(score * 10 )  + ' / 10 ! 🤗'
            elif score > 0.3:
                return '¯\_(ツ)_/¯ I’ll rate it a ' + str(score * 10) +  ' /10. Nice try!'
            return 'Not sure about that one…  (.-.)'



if __name__ == "__main__":
    #init(DEFAULT_MODEL_PATH, DEVICE_JSON)
    #print("ready")
    #app.run(host='192.168.1.10')
    #app.run(host='192.168.1.10', port=4444)
    app.run(port=5000)
    #app.run(host='192.168.1.10', port=4444, ssl_context=('/home/tobias/fullchain1.pem', '/home/tobias/privkey1.pem'))


