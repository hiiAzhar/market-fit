from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and preprocessing
model = joblib.load('model/market_fit_model.pkl')
preprocessing = joblib.load('model/preprocessing.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Convert numerical fields to float
        num_fields = ['Price', 'Discount', 'Tax Rate', 'Stock Level', 
                     'Shipping Cost', 'Return Rate']
        for field in num_fields:
            data[field] = float(data[field])
        
        # Create DataFrame and feature engineering
        input_df = pd.DataFrame([data])
        input_df['Price_Discount_Ratio'] = input_df['Price'] / (input_df['Discount'] + 1)
        input_df['Stock_Shipping_Ratio'] = input_df['Stock Level'] / (input_df['Shipping Cost'] + 1)
        input_df['Price_Tax_Ratio'] = input_df['Price'] / (input_df['Tax Rate'] + 1)
        input_df['Stock_Return_Ratio'] = input_df['Stock Level'] / (input_df['Return Rate'] + 1)

        # Encoding and preprocessing
        input_encoded = pd.get_dummies(input_df, columns=preprocessing['categorical_features'])
        for col in preprocessing['top_features']:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[preprocessing['top_features']]
        
        # Scaling
        num_features = preprocessing['scaler'].feature_names_in_
        input_encoded[num_features] = preprocessing['scaler'].transform(
            input_encoded[num_features] + 1e-6
        )

        # Prediction with type conversion
        proba = model.predict_proba(input_encoded)[0][1]
        prediction = model.predict(input_encoded)[0]

        return jsonify({
            'prediction': int(prediction),  # Convert numpy int to Python int
            'probability': float(round(proba * 100, 1)),  # Convert numpy float to Python float
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'})

if __name__ == '__main__':
    app.run(debug=True)