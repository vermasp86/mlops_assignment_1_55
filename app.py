import mlflow
from flask import Flask, request, jsonify

# 1. Model Loading
# Best model ka naam ya URI yahan set karte hain.
# Assume ki "LogisticRegression_Heart_Classifier" hamara final registered model hai
MODEL_NAME = "LogisticRegression_Heart_Classifier" 
MODEL_URI = f"models:/{MODEL_NAME}/latest" 

try:
    # Model ko MLflow Registry se load karo
    loaded_model = mlflow.sklearn.load_model(MODEL_URI)
    print(f"--- Model '{MODEL_NAME}' successfully loaded from MLflow ---")
except Exception as e:
    print(f"Error loading model from MLflow: {e}")
    # Agar model load nahi ho paya, toh application fail kar do
    raise

# 2. Flask Application Setup
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to receive patient data and return heart disease prediction."""
    try:
        # Request body se JSON data lo
        data = request.get_json(force=True)
        
        # Data ko DataFrame mein convert karo (model input ke liye zaroori)
        # Note: Input data structure wahi hona chahiye jo training data mein tha
        import pandas as pd
        input_df = pd.DataFrame([data])
        
        # Model prediction karo
        prediction = loaded_model.predict(input_df)
        
        # Prediction output 0 ya 1 hoga.
        result = int(prediction[0])

        return jsonify({
            'prediction': result,
            'status': 'Prediction successful'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'Prediction failed'
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'model': MODEL_NAME})


if __name__ == '__main__':
    # Production ke liye, host ko 0.0.0.0 set karo taki Docker access kar sake
    app.run(host='0.0.0.0', port=5000)