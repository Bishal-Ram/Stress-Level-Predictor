from flask import Flask,request,jsonify
import numpy as np
import pickle

model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict',methods=['POST'])
def predict():
    body_temperature = request.form.get('body_temperature')
    pulse = request.form.get('pulse')
    spo2 = request.form.get('spo2')

    input_query = np.array([[body_temperature,pulse,spo2]],dtype=np.float64)

    result = model.predict(input_query)[0]

    return jsonify({'state':str(result)})

if __name__ == '__main__':
    app.run(debug=True)
