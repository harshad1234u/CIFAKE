import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras import layers, models, callbacks

# ==========================================
# CONFIGURATION
# ==========================================
# Replace this with the path to the extracted Kaggle dataset
# It assumes a folder structure like:
# dataset_path/
# ├── real/
# └── fake/
DATASET_PATH = "dataset_stylegan" 

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

def build_robust_model():
    """
    Builds a robust classifier using EfficientNetV2B0, which is excellent
    for capturing subtle, high-frequency artifacts in Deepfakes.
    """
    # 1. Base Model (Transfer Learning)
    # EfficientNetV2 includes its own preprocessing, so we don't need Rescaling.
    base_model = EfficientNetV2B0(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False # Freeze base model initially

    # 2. Data Augmentation (Crucial for faces/deepfakes to prevent overfitting)
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")

    # 3. Assemble Model
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset path '{DATASET_PATH}' not found.")
        print("Please download the StyleGAN dataset from Kaggle, extract it, and place 'real' and 'fake' folders inside.")
        return

    print("Loading Dataset...")
    
    # 80% Training
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # 20% Validation
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Optimize datasets for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

    model = build_robust_model()
    model.summary()

    # Setup Callbacks
    # EarlyStopping: Stops training if the model stops improving
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True
    )
    
    # ModelCheckpoint: Saves the best model during training
    checkpoint = callbacks.ModelCheckpoint(
        filepath='stylegan_detector_best.keras',
        monitor='val_accuracy',
        save_best_only=True
    )

    # ReduceLROnPlateau: Lowers learning rate when validation loss plateaus
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=2, 
        min_lr=1e-6
    )

    print("Starting Training Phase...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[early_stopping, checkpoint, reduce_lr]
    )

    # Save final model
    model.save('stylegan_detector_final.keras')
    print("Training Complete! Models saved as 'stylegan_detector_best.keras' and 'stylegan_detector_final.keras'")

if __name__ == "__main__":
    main()
