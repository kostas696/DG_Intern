

from flask import Flask, render_template, request
from flask import jsonify
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Trying to unpickle estimator")

app = Flask(__name__)

# Set the static folder path
app.config['STATIC_FOLDER'] = 'static'

# Load the scaler
with open(r'C:\Users\User\Desktop\depl_flask\scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Load the model
with open(r'C:\Users\User\Desktop\depl_flask\gradient_boosting_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the prediction route
@app.route('/', methods=['GET'])
def index():
   return render_template('index.html')
   
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the form data
        property_type = request.form['property_type']
        neighborhood = request.form['neighborhood']
        property_sqft = float(request.form['property_sqft'])
        bedrooms = int(request.form['bedrooms'])
        baths = int(request.form['baths'])

        # Load the column names of the encoded features
        with open('encoded_columns.pkl', 'rb') as f:
          encoded_columns = pickle.load(f)

        # Perform one-hot encoding for categorical variables
        type_encoded = pd.get_dummies(pd.Series(property_type)).reindex(columns=encoded_columns, fill_value=0)
        sublocality_encoded = pd.get_dummies(pd.Series(neighborhood)).reindex(columns=encoded_columns, fill_value=0)

        # Concatenate encoded features
        encoded_features = pd.concat([type_encoded, sublocality_encoded], axis=1)

        # Scale the entire feature vector (both numerical and encoded categorical features)
        input_features = np.array([[property_sqft, bedrooms, baths]])  # Numerical features
        input_features_scaled = scaler.transform(np.concatenate([input_features, encoded_features.values], axis=1))

        # Make the prediction
        prediction = model.predict(input_features_scaled)
        output = round(prediction[0], 2)
        print(prediction) 

        # Display the prediction result
        return jsonify({"prediction_text": "House price should be ${} ".format(output)})

    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=5000, debug=False)
