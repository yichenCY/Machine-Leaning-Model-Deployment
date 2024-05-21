from flask import Flask,  render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('Final_Model.pkl')

# render default webpage
# Get request
@app.route('/',methods =['GET'])
def home():
    return render_template('web_structure.html')

# Post prediction output
@app.route('/', methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('web_structure.html', prediction=output)

if __name__ == '__main__':
    app.run(debug=True)
