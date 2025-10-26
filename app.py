import os
import io
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
from PIL import Image
import logging
# Removed Gemini import: import google.generativeai as genai
import mimetypes

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# --- Configuration ---
VGG_MODEL_FILENAME = 'best_model_vgg16_finetuned (1).h5' # Your main prediction model
DETECTOR_MODEL_FILENAME = 'chest_xray_detector.keras' # Your new detector model
VGG_IMAGE_SIZE = (224, 224)
# --- FIX 1: Set correct detector image size ---
DETECTOR_IMAGE_SIZE = (128, 128)
UPLOAD_FOLDER = 'uploads' # Optional

# --- Load the Keras Models ---
vgg_model = None
detector_model = None

# Load VGG Model (Pneumonia Detection)
try:
    if os.path.exists(VGG_MODEL_FILENAME):
        logging.info(f"Loading VGG model from {VGG_MODEL_FILENAME}...")
        vgg_model = tf.keras.models.load_model(VGG_MODEL_FILENAME)
        # Warm up the model
        dummy_input_vgg = np.zeros((1, VGG_IMAGE_SIZE[0], VGG_IMAGE_SIZE[1], 3), dtype=np.float32)
        vgg_model.predict(dummy_input_vgg)
        logging.info("VGG model loaded and warmed up successfully.")
    else:
        logging.error(f"VGG model file not found at {VGG_MODEL_FILENAME}. Prediction endpoint will not work.")
except Exception as e:
    logging.error(f"Error loading VGG model: {e}")
    vgg_model = None

# Load Detector Model (X-ray vs Not X-ray)
try:
    if os.path.exists(DETECTOR_MODEL_FILENAME):
        logging.info(f"Loading Detector model from {DETECTOR_MODEL_FILENAME}...")
        detector_model = tf.keras.models.load_model(DETECTOR_MODEL_FILENAME)
         # Warm up the detector model with the CORRECT size
        dummy_input_detector = np.zeros((1, DETECTOR_IMAGE_SIZE[0], DETECTOR_IMAGE_SIZE[1], 3), dtype=np.float32)
        # Note: We don't divide by 255 here, assuming the model handles it internally like predict.py
        detector_model.predict(dummy_input_detector)
        logging.info("Detector model loaded and warmed up successfully.")
    else:
        logging.warning(f"Detector model file not found at {DETECTOR_MODEL_FILENAME}. Image type check will be skipped.")
except Exception as e:
    # Log the specific error during loading
    logging.error(f"Error loading Detector model: {e}", exc_info=True)
    detector_model = None


# --- Preprocessing Functions ---

def preprocess_for_detector(image_bytes):
    """Preprocesses image for the detector model."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize(DETECTOR_IMAGE_SIZE, Image.NEAREST) # Use CORRECT detector size
        img_array = np.array(img)
        # --- FIX 2: Remove scaling, assuming model has internal Rescaling layer ---
        # img_tensor = tf.cast(img_array, tf.float32) / 255.0
        img_tensor = tf.cast(img_array, tf.float32) # Cast only
        img_batch = tf.expand_dims(img_tensor, axis=0)
        return img_batch
    except Exception as e:
        logging.error(f"Error preprocessing image for detector: {e}")
        return None

def preprocess_for_vgg(image_bytes):
    """Preprocesses image for the VGG prediction model."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize(VGG_IMAGE_SIZE, Image.NEAREST) # Use VGG size
        img_array = np.array(img)
        # Simple scaling (matching VGG training)
        img_tensor = tf.cast(img_array, tf.float32) / 255.0
        img_batch = tf.expand_dims(img_tensor, axis=0)
        return img_batch
    except Exception as e:
        logging.error(f"Error preprocessing image for VGG: {e}")
        return None

# --- Image Type Check using Detector Model ---
def check_image_type(image_bytes):
    """
    Uses the loaded detector model to classify if the image is a chest X-ray.
    Returns True if likely a chest X-ray, False otherwise, None if detector not loaded or error.
    """
    if detector_model is None:
        logging.warning("Detector model not loaded, skipping image type check.")
        return None # Indicate check couldn't be performed

    processed_image = preprocess_for_detector(image_bytes)
    if processed_image is None:
        logging.error("Preprocessing for detector failed.")
        return None # Preprocessing failed

    try:
        logging.info("Running detector model prediction...")
        # Add verbose=0 to suppress Keras progress bar noise in logs
        prediction = detector_model.predict(processed_image, verbose=0)
        probability = prediction[0][0] # Assuming single output neuron
        logging.info(f"Detector model raw output: {probability}")

        # --- FIX 3: Interpret output based on predict.py logic ---
        # score < 0.5 means chest_xray (class 0)
        is_xray = probability < 0.5
        logging.info(f"Detector model classified as chest X-ray: {is_xray}")
        return is_xray

    except Exception as e:
        logging.error(f"Error during detector model prediction: {e}", exc_info=True)
        return None # Indicate an error occurred

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

# Add HEAD method support for the status check
@app.route('/predict', methods=['POST', 'HEAD'])
def predict():
    """Handles image upload, content check, and prediction."""
    if request.method == 'HEAD':
        # Check if BOTH models are loaded for full readiness
        if vgg_model and detector_model:
            return '', 200 # OK
        elif vgg_model:
             logging.warning("Backend check: VGG model loaded, but Detector model is missing.")
             # Consider returning 503 if detector is essential, or 200 if optional
             return '', 200 # Let's say OK, but check won't work
        else:
            logging.error("Backend check: VGG model not loaded.")
            return '', 503 # Service Unavailable if main VGG model isn't loaded

    # Handle POST request for prediction
    if vgg_model is None:
        logging.error("Prediction attempted but VGG model is not loaded.")
        return jsonify({'error': 'Main prediction model not loaded. Cannot perform prediction.'}), 503

    if 'file' not in request.files:
        logging.warning("No file part in POST request.")
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        logging.warning("No selected file in POST request.")
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            logging.info(f"Received file: {file.filename}")
            # Read bytes ONCE and store
            try:
                 image_bytes = file.read()
                 # Reset stream position in case it's needed again (though not in current flow)
                 # file.stream.seek(0)
            except Exception as e:
                 logging.error(f"Error reading file bytes: {e}")
                 return jsonify({'error': 'Could not read image file.'}), 500


            # --- Check image type using the detector model ---
            is_xray_result = check_image_type(image_bytes)

            if is_xray_result is None:
                 # Error during check, or detector model not loaded
                 logging.warning("Could not verify image type using detector model. Proceeding with VGG prediction anyway.")
                 # No 'pass' needed, code flow continues naturally
            elif not is_xray_result:
                logging.info(f"Image '{file.filename}' classified as NOT a chest X-ray by detector model.")
                return jsonify({'error': 'The uploaded image was classified as not being a chest X-ray. Please upload a valid chest X-ray.'}), 400
            # --- End of Detector Check ---

            logging.info(f"Image '{file.filename}' check passed or skipped. Proceeding with VGG prediction.")
            # Preprocess the image for VGG using the stored bytes
            processed_image_vgg = preprocess_for_vgg(image_bytes)
            if processed_image_vgg is None:
                 logging.error("Preprocessing for VGG failed.")
                 return jsonify({'error': 'Failed to preprocess image for VGG model'}), 500

            # Make prediction using VGG model
            logging.info("Running VGG model prediction...")
            # Add verbose=0 here too
            prediction = vgg_model.predict(processed_image_vgg, verbose=0)
            probability = prediction[0][0]
            logging.info(f"VGG Raw prediction probability: {probability}")

            if probability > 0.5:
                label = 'PNEUMONIA DETECTED'
                confidence = probability * 100
            else:
                label = 'NO PNEUMONIA'
                confidence = (1 - probability) * 100

            logging.info(f"Prediction: {label}, Confidence: {confidence:.2f}%")

            return jsonify({
                'prediction': label,
                'confidence': round(confidence, 2)
            })

        except Exception as e:
            logging.error(f"Error during prediction process: {e}", exc_info=True)
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500

    logging.error("Reached end of predict function without returning, likely an issue with file handling.")
    return jsonify({'error': 'Unknown server error during file processing'}), 500

# Optional: Favicon route
@app.route('/favicon.ico')
def favicon():
    return '', 404

if __name__ == '__main__':
    # Make sure to place chest_xray_detector.keras in the same directory
    app.run(host='0.0.0.0', port=5000, debug=False)

