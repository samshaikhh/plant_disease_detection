"""
train.py
--------
Trains the plant disease classification model.

Usage:
    python train.py --epochs 15 --batch_size 32 --lr 0.0001
    python train.py --epochs 5 --fine_tune True   # second phase fine-tuning
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from data_utils import get_datasets, IMG_SIZE
from build_model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train plant disease detection model")
    parser.add_argument("--data_dir", type=str, default="data/split",
                         help="Path to split dataset (train/val/test folders)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fine_tune", type=bool, default=False,
                         help="Unfreeze base model layers for fine-tuning")
    parser.add_argument("--output_dir", type=str, default="../saved_model")
    return parser.parse_args()


def plot_history(history, output_dir):
    """Plots and saves training/validation accuracy and loss curves."""
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Accuracy")
    plt.xlabel("Epoch")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Loss")
    plt.xlabel("Epoch")

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(plot_path)
    print(f"Saved accuracy/loss plot to {plot_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading datasets...")
    train_ds, val_ds, test_ds, class_names = get_datasets(
        data_dir=args.data_dir, img_size=IMG_SIZE, batch_size=args.batch_size
    )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # Save class names so the backend can map prediction index -> label
    with open(os.path.join(args.output_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    print("Building model...")
    model = build_model(num_classes, img_size=IMG_SIZE, fine_tune=args.fine_tune)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Callbacks: stop early if val_loss stops improving, save best model
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.output_dir, "best_model.keras"),
            monitor="val_accuracy", save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]

    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Save final model (Keras native format, recommended over .h5)
    final_path = os.path.join(args.output_dir, "plant_disease_model.keras")
    model.save(final_path)
    print(f"Model saved to {final_path}")

    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump({"test_accuracy": test_acc, "test_loss": test_loss}, f, indent=2)

    plot_history(history, args.output_dir)


if __name__ == "__main__":
    main()
