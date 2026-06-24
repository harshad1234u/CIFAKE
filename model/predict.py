import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Dictionaries to store loaded models and their target input sizes (singleton style)
_models = {}
_target_sizes = {}

def get_model(model_name="CIFAKE (MobileNetV2)"):
    """
    Singleton pattern to load the specified model only once.
    """
    global _models, _target_sizes
    
    if model_name not in _models:
        # Determine the file path of the model
        if "StyleGAN" in model_name:
            # Check root directory or model directory
            model_path_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'stylegan_detector_best.keras')
            model_path_folder = os.path.join(os.path.dirname(__file__), 'stylegan_detector_best.keras')
            
            if os.path.exists(model_path_root):
                model_path = model_path_root
            elif os.path.exists(model_path_folder):
                model_path = model_path_folder
            else:
                raise FileNotFoundError(f"StyleGAN model file not found in root or model directory.")
        else:
            model_path = os.path.join(os.path.dirname(__file__), 'cifake_model.keras')
            if not os.path.exists(model_path):
                raise FileNotFoundError("CIFAKE model file not found.")
            
        try:
            _models[model_name] = tf.keras.models.load_model(model_path)
            input_shape = _models[model_name].input_shape
            _target_sizes[model_name] = (input_shape[1], input_shape[2])
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")
            
    return _models[model_name], _target_sizes[model_name]

def predict_image(image, model_name="CIFAKE (MobileNetV2)"):
    """
    Predicts whether an image is REAL or FAKE using the specified model.
    Takes either a file path or a PIL Image object.
    
    Returns:
        dict: {'label': str, 'confidence': float, 'probability': float}
    """
    model, target_size = get_model(model_name)
    
    try:
        if isinstance(image, str):
            img = Image.open(image)
        else:
            img = image
            
        img = img.convert('RGB')
        img = img.resize(target_size)
        
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array, verbose=0)
        probability = float(prediction[0][0])
        
        is_real = probability > 0.5
        label = "REAL" if is_real else "AI Generated"
        
        confidence = probability if is_real else 1.0 - probability
        
        return {
            'label': label,
            'confidence': confidence,
            'probability': probability
        }
    except Exception as e:
        raise ValueError("Image processing failed or corrupted image.")

