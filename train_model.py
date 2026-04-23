"""
VigiDrive - CNN Model Training Script
Trains a lightweight CNN on eye-crop images to classify:
  0 = Open Eyes
  1 = Closed Eyes

Uses the CEW (Closed Eyes in the Wild) or MRL Eye Dataset.
Saves model to models/eye_classifier.h5

Usage:
    python train_model.py --data_dir data/eye_dataset --epochs 30
"""

import cv2
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

import os
import argparse
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[ERROR] TensorFlow not found. Install: pip install tensorflow")


def build_model(input_shape=(24, 24, 1)) -> "keras.Model":
    """
    Lightweight CNN for eye-state classification.
    Designed to run in real-time on CPU.
    """
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.4),

        # Dense
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(2, activation="softmax"),   # [open, closed]
    ], name="VigiDrive_EyeCNN")
    return model


def build_data_generators(data_dir: str, img_size: int = 24,
                           batch_size: int = 32):
    """
    Expects directory structure:
        data_dir/
            open/    ← open-eye images
            closed/  ← closed-eye images
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
    )

    train = train_gen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )
    val = train_gen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )
    return train, val


def train(data_dir: str, epochs: int = 30, batch_size: int = 32,
          img_size: int = 24, output: str = "models/eye_classifier.h5"):
    if not TF_AVAILABLE:
        return

    print(f"[INFO] Data dir   : {data_dir}")
    print(f"[INFO] Epochs     : {epochs}")
    print(f"[INFO] Image size : {img_size}x{img_size}")

    os.makedirs(os.path.dirname(output), exist_ok=True)

    train_data, val_data = build_data_generators(data_dir, img_size, batch_size)
    print(f"[INFO] Classes    : {train_data.class_indices}")

    model = build_model(input_shape=(img_size, img_size, 1))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            output, save_best_only=True, monitor="val_accuracy", verbose=1),
        keras.callbacks.EarlyStopping(
            patience=7, restore_best_weights=True, monitor="val_accuracy"),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        keras.callbacks.TensorBoard(log_dir="logs/tensorboard"),
    ]

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks,
    )

    # Evaluate
    loss, acc = model.evaluate(val_data)
    print(f"\n[RESULT] Validation Accuracy : {acc*100:.2f}%")
    print(f"[RESULT] Model saved to      : {output}")
    return history


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data/eye_dataset")
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--img_size",   type=int, default=24)
    p.add_argument("--output",     default="models/eye_classifier.h5")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data_dir, args.epochs, args.batch_size,
          args.img_size, args.output)
