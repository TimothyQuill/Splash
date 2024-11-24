# Splash

### Install requirements
`cd /path/to/repo/Splash` \
`pip install -r requirements.txt`

## Steps for training

### 1. Adjust hyperparameters
All the values to train the model can be found in `hyperparamters.py`, and are set to what was used for training.

### 2. Train the model
`python train.py`

## Steps for inference

### Run the Flask Application
`python app.py`

### 5. Test the API Endpoint
Test the `/generate` endpoint using a tool like cURL.
e.g. `curl -X POST http://127.0.0.1:5000/generate \
     -H "Content-Type: application/json" \
     -d '[INPUT_DATA]'`