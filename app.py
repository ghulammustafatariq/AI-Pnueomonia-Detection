import os
import io
import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
import logging
# --- FIX: Import the module directly for robustness ---
from tensorflow.keras.applications import mobilenet_v2 # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# --- Configuration ---
VGG_MODEL_FILENAME = 'best_model_vgg16_finetuned (1).h5' # Your main prediction model
DETECTOR_MODEL_FILENAME = 'chest_xray_detector_mobilenet (1).keras' # Your new detector model
VGG_IMAGE_SIZE = (224, 224)
DETECTOR_IMAGE_SIZE = (128, 128) # Correct size for the detector

# --- Helper function to build the detector model architecture ---
def create_detector_model():
    """Creates the exact MobileNetV2 architecture used during training."""
    base_model = mobilenet_v2.MobileNetV2(
        input_shape=(DETECTOR_IMAGE_SIZE[0], DETECTOR_IMAGE_SIZE[1], 3),
        include_top=False,
        weights=None # We only need the architecture, will load weights from file
    )
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ], name="chest_xray_detector") # Give it a name
    return model

# --- Load the Keras Models ---
vgg_model = None
detector_model = None

# Load VGG Model (Pneumonia Detection)
try:
    if os.path.exists(VGG_MODEL_FILENAME):
        logging.info(f"Loading VGG model from {VGG_MODEL_FILENAME}...")
        # --- FIX: Add compile=False to skip optimizer loading ---
        vgg_model = tf.keras.models.load_model(VGG_MODEL_FILENAME, compile=False)
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
        
        # --- FIX: Rebuild model architecture and load weights ---
        # This bypasses graph-loading errors from version mismatches.
        
        # 1. Create the model structure
        detector_model = create_detector_model()

        # 2. Build the model by calling it with dummy data
        # (This is necessary before loading weights)
        dummy_input_detector = np.zeros((1, DETECTOR_IMAGE_SIZE[0], DETECTOR_IMAGE_SIZE[1], 3), dtype=np.float32)
        dummy_input_detector_preprocessed = mobilenet_v2.preprocess_input(dummy_input_detector)
        detector_model(dummy_input_detector_preprocessed) # This builds the model
        
        # 3. Load *only* the weights from the .keras file
        detector_model.load_weights(DETECTOR_MODEL_FILENAME)
        
        # 4. Do the warm-up predict
        detector_model.predict(dummy_input_detector_preprocessed)
        # --- END FIX ---

        logging.info("Detector model loaded and warmed up successfully.")
    else:
        logging.warning(f"Detector model file not found at {DETECTOR_MODEL_FILENAME}. Image type check will be skipped.")
except Exception as e:
    # Log the specific error during loading
    logging.error(f"Error loading Detector model: {e}", exc_info=True)
    detector_model = None


# --- Preprocessing Functions ---

def preprocess_for_detector(image_bytes):
    """Preprocesses image for the detector model (MobileNetV2)."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        # Use BICUBIC for better resize quality
        img = img.resize(DETECTOR_IMAGE_SIZE, Image.BICUBIC) 
        img_array = np.array(img)
        img_batch = tf.expand_dims(img_array, axis=0)
        
        # --- CRITICAL FIX: Cast to float32 BEFORE preprocessing ---
        img_batch_float = tf.cast(img_batch, tf.float32)
        
        # Now preprocess the float32 tensor
        img_preprocessed = mobilenet_v2.preprocess_input(img_batch_float)
        return img_preprocessed
        # --- END FIX ---

    except Exception as e:
        logging.error(f"Error preprocessing image for detector: {e}")
        return None

def preprocess_for_vgg(image_bytes):
    """Preprocesses image for the VGG prediction model."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        # Use BICUBIC for better resize quality
        img = img.resize(VGG_IMAGE_SIZE, Image.BICUBIC)
        img_array = np.array(img)
        # Simple scaling (assuming this is how your VGG model was trained)
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
        prediction = detector_model.predict(processed_image, verbose=0)
        probability = prediction[0][0] # Assuming single output neuron
        logging.info(f"Detector model raw output: {probability}")

        # --- CORRECTED LOGIC (Your FIX 3) ---
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
    # This will look for 'index.html' in a folder named 'templates'
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
            return '', 200 # OK, but check won't work
        else:
            logging.error("Backend check: VGG model not loaded.")
            return '', 503 # Service Unavailable

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
            image_bytes = file.read()
            
            # --- Check image type using the detector model ---
            is_xray_result = check_image_type(image_bytes)

            if is_xray_result is None:
                # Error during check, or detector model not loaded
                logging.warning("Could not verify image type. Proceeding with prediction anyway.")
            elif not is_xray_result:
                logging.info(f"Image '{file.filename}' classified as NOT a chest X-ray.")
                # Return a specific error message to the user
                return jsonify({'error': 'This does not appear to be a chest X-ray. Please upload a valid image.'}), 400
            # --- End of Detector Check ---

            logging.info(f"Image '{file.filename}' check passed or skipped. Proceeding with VGG prediction.")
            processed_image_vgg = preprocess_for_vgg(image_bytes)
            if processed_image_vgg is None:
                logging.error("Preprocessing for VGG failed.")
                return jsonify({'error': 'Failed to preprocess image for VGG model'}), 500

            # Make prediction using VGG model
            logging.info("Running VGG model prediction...")
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

    logging.error("Reached end of predict function without returning.")
    return jsonify({'error': 'Unknown server error during file processing'}), 500

# Optional: Favicon route
@app.route('/favicon.ico')
def favicon():
    return '', 404

if __name__ == '__main__':
    # Make sure models are in the same directory as app.py
    # Gunicorn/Waitress would be used for production instead of app.run
    app.run(host='0.0.0.0', port=5000, debug=False)

