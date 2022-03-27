from flask import Flask, jsonify, request
import pickle
import os
import traceback

app = Flask(__name__)

@app.route("/", methods=['GET'])
def hello():
    return "hey"

@app.route('/predict', methods=['POST'])
def predict():
	try:
		json = request.get_json()	 
		temp=list(json[0].values())
		rf = pickle.load(open('rf_baseline.pkl', 'rb'))
		prediction = rf.predict_proba([temp])
		print("Prediction: ", prediction)        
		return jsonify({'prediction': str(prediction[0][1]*100)})

	except:        
		return jsonify({'trace': traceback.format_exc()})
    


if __name__ == '__main__':
    app.run(debug=True)
