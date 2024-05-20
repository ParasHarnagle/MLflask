import pickle

import uvicorn #ASGI
from fastapi import FastAPI
from banknote import BankNote

app = FastAPI()

@app.get('/')
def index():
    return {'message':"hello world model deployemnt"}

pickle_in = open("classifier.pkl","rb")
classifier = pickle.load(pickle_in)

@app.post('/predict')
def predict_banknotes(data:BankNote):
    data = data.dict()
    print(data)
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['curtosis']
    predict = classifier.predict([[variance, skewness,curtosis,entropy]])
    print(f"prediction {predict}")
    if predict[0]> 0.5:
        prediction = "Fake Note"
    else:
        prediction = "Valid Note"
    return {
        'prediciton':prediction
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1",port=8000)
