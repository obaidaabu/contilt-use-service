from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS
import spacy_universal_sentence_encoder
from analysis.descriptionphrases import DescriptionPhrases
from analysis.summarization import Summarization

nlp = spacy_universal_sentence_encoder.load_model('en_use_lg')
summarizer = Summarization(nlp)
#spacy_model = spacy.load('en_core_web_lg')
app = Flask("test")
cors = CORS(app, resources={r"/*": {"origins": "*"}})
descriptionPhrases = DescriptionPhrases()

@app.route('/api/getsemanticvectors', methods=['POST'])
def getSemanticVectors():
    print("creating semantic vector")
    body = request.get_json()
    res = {}
    for sentence in body["sentences"]:
        sent1 = nlp(sentence, disable=["parser", "ner"])
        sent_vec1 = sent1.vector.tolist()
        res[sentence] = sent_vec1
    return jsonify(res)

@app.route('/api/getSimilarityToQuery', methods=['POST'])
def getSimilarityToQuery():
    print("creating semantic vector")
    body = request.get_json()
    query = body["query"]
    res = {}
    query_vec = nlp(query, disable=["parser", "ner"])
    for sentence in body["sentences"]:
        sent_vec = nlp(sentence, disable=["parser", "ner"])
        res[sentence] = query_vec.similarity(sent_vec)
    return jsonify(res)

# receives a list of items, each with "text" and "score" fields
@app.route('/api/getDescriptivePhrasesSummarized', methods=['POST'])
def getDescriptivePhrasesSummarized():
    data = request.get_json()
    phrases_to_weights = descriptionPhrases.extract(data)
    res = summarizer.weightedSummary(phrases_to_weights, 50)
    return jsonify(res)

# receives a map of sentences to scores (one object that keys are sentences and values are their weights)
@app.route('/api/summarizeWeighted', methods=['POST'])
def summarizeWeighted():
    data = request.get_json()
    res = summarizer.weightedSummary(data)
    return jsonify(res)

@app.route('/api/semanticAnalysis', methods=['POST'])
def semanticAnalysis():
    data = request.get_json()
    res = descriptionPhrases.extract(data)
    return jsonify(res)

@app.route('/api/health', methods=['GET'])
def health():

    return jsonify("OK")


app.run(host='0.0.0.0', port=5002, threaded=True)
