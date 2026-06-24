import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Generates a Grad-CAM heatmap for the given image array and model.
    """
    # Find the global average pooling layer dynamically
    gap_layer = None
    for layer in model.layers:
        if "global_average_pooling2d" in layer.name.lower():
            gap_layer = layer
            break
            
    if gap_layer is None:
        raise ValueError("Could not find global_average_pooling2d layer in the model architecture.")
        
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [gap_layer.input, model.output]
    )

    # 2. Compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            # We are doing binary classification, so preds is just a single probability
            # The "score" we want to maximize is the probability of the predicted class.
            # If preds > 0.5, we want to explain why it's REAL (class 1).
            # If preds <= 0.5, we want to explain why it's FAKE (class 0).
            # For class 1, maximize preds. For class 0, maximize (1 - preds).
            pred_value = preds[0][0]
            if pred_value > 0.5:
                class_channel = preds
            else:
                class_channel = 1.0 - preds
        else:
            class_channel = preds

    # 3. This is the gradient of the output neuron with respect to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 4. Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Multiply each channel in the feature map array by "how important this channel is"
    last_conv_layer_output = last_conv_layer_output[0]
    # We add an axis to pooled_grads to broadcast
    pooled_grads = pooled_grads[..., tf.newaxis]
    heatmap = last_conv_layer_output @ pooled_grads
    heatmap = tf.squeeze(heatmap)

    # 6. Normalize the heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, alpha=0.4):
    """
    Superimposes the heatmap over the original image.
    Returns:
        tuple of PIL Images: (original_image, heatmap_image, superimposed_image)
    """
    # Load original image
    img = tf.keras.utils.load_img(img_path)
    img = tf.keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # OpenCV uses BGR, convert to RGB
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)

    # Resize jet heatmap to match image size
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))

    # Superimpose the heatmap on original image
    superimposed_img = jet * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
    
    # Heatmap only
    heatmap_img = tf.keras.utils.array_to_img(jet)
    
    # Original image
    original_img = tf.keras.utils.array_to_img(img)

    return original_img, heatmap_img, superimposed_img

def generate_gradcam(image_path, model, target_size):
    """
    Complete pipeline to generate Grad-CAM for a given image and model.
    """
    # Prepare image
    img = Image.open(image_path).convert('RGB').resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Generate heatmap
    heatmap = make_gradcam_heatmap(img_array, model)
    
    # Create overlays
    original, heatmap_img, overlay = save_and_display_gradcam(image_path, heatmap)
    
    return original, heatmap_img, overlay
