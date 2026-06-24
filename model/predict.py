import os
import numpy as np
import tensorflow as tf
from PIL import Image

_model = None
_target_size = None

def get_model():
    """Singleton pattern to load the model only once."""
    global _model, _target_size
    if _model is None:
        model_path = os.path.join(os.path.dirname(__file__), 'cifake_model.keras')
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model unavailable.")
            
        try:
            _model = tf.keras.models.load_model(model_path)
            input_shape = _model.input_shape
            _target_size = (input_shape[1], input_shape[2])
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")
            
    return _model, _target_size

def predict_image(image):
    """
    Predicts whether an image is REAL or FAKE.
    Takes either a file path or a PIL Image object.
    
    Returns:
        dict: {'label': str, 'confidence': float, 'probability': float}
    """
    model, target_size = get_model()
    
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
