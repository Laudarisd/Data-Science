from numpy.lib.twodim_base import tri
#from lstm_anomaly import AutoEncoder
import pandas as pd
import numpy as np
import flask
from tensorflow.keras.models import load_model
import joblib
import csv
import codecs
import warnings




def warn(*arg, **kwargs):
    pass

warnings.warn = warn

#initialize the flask application

app = flask.Flask(__name__)

#load the pre-trained model

def define_model():
    global model
    model = load_model('./model/anomaly_model.h5')
    return print("Model is loaded")


limit = 11

# this method process any requests to the  submit endpoint

@app.route("/", methods=["GET","POST"])

def submit():
    #initialize the data dictionary that will be returned in the response
    data_out = {}
    #load the data file from our endpoint
    if flask.request.method == "POST":
        #read the data file
        file = flask.request.files['data_file']
        if not file:
            return "No file submitted"
        data = []
        stream, = codecs.iterdecode(file.stream, 'utf-8')
        for row in csv.reader(stream, dialect=csv.excel):
            if row:
                data.append(row)
                
        #convert input data to pandas dataframe
        
        df = pd.DataFrame(data)
        df.set_index(df.iloc[:, 0], inplace=True)
        df2 = df.drop(df.columns[0], axis=1)
        df2 = df2.astype(np.float64)
        print(df2)
        
        #normalize the data
        scaler = joblib.load('./data/combined.csv')
        X = scaler.transform(df2)
        #reshape data set for LSTM [sample, time steps, features]
        X = X.reshape(X.shape[0], 1, X.shape[1])
    #calculate the reconstruction loss on the input data
    
        data_out['Analysis'] = []
        preds = model.predict(X)
        preds = preds.reshape(preds.shape[0], preds.shape[2])
        preds = pd.DataFrame(preds, columns=df2.columns)
        preds.index = df2.index
        
        scored = pd.DataFrame(index=df2.index)
        yhat = X.reshape(X.shape[0], X.reshape[2])
        scored['Loss_mae'] = np.mean(np.abs(yhat - preds), axis=1)
        scored['Threshold'] = limit
        scored['Anomaly'] = scored['Loss_mae'] > scored['threshold']
        print(scored)
    
        #determine of an anomaly was detected
        
        triggered = []
        for i in range(len(scored)):
            temp = scored.iloc[i]
            if temp.iloc[2]:
                triggered.append(temp)
        print(len(triggered))
        if len(triggered) > 0:
            for j in range(len(triggered)):
                out = triggered[j]
                result = {"Anomaly": True, "Value":out[0], "filename":out.name} 
                data_out["Analysis"].append(result)
        else:
            result = {"Anomaly":"No Anomalies Detected"}
            data_out["Analysis"].append(result)
    print(data_out)
        
    return flask.jsonify(data_out)



#first load model and start the server
#we need to specify the host of 0.0.0 so the app is available on both localhost as well as 
# as on the external IP of the Docker container


if __name__ == "__main__":
    print(("* Loading the Keras model and starting the server ...."
          "Please wait until the server has fully started before submitting"))

    define_model()
    #app.run(host='0.0.0.0')
    app.run(debug=True)