"""Fine-tune the Teachable Machine classifier on augmented image samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:
    import tf_keras as keras
except ImportError as exc:  # pragma: no cover - friendly setup error
    raise SystemExit(
        "Missing dependency. Run: pip install -r requirements.txt"
    ) from exc


IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Cat", "Dog"]


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        fitted = ImageOps.fit(
            image.convert("RGB"), IMAGE_SIZE, method=Image.Resampling.LANCZOS
        )
    return (np.asarray(fitted, dtype=np.float32) / 127.5) - 1.0


def load_split(samples_dir: Path, validation_count: int) -> tuple[np.ndarray, ...]:
    train_x: list[np.ndarray] = []
    train_y: list[int] = []
    validation_x: list[np.ndarray] = []
    validation_y: list[int] = []

    for label, class_name in enumerate(CLASS_NAMES):
        files = sorted((samples_dir / class_name).glob("*.jpg"))
        if len(files) <= validation_count:
            raise ValueError(
                f"{class_name} needs more than {validation_count} samples; found {len(files)}"
            )
        split_at = len(files) - validation_count
        for path in files[:split_at]:
            train_x.append(load_image(path))
            train_y.append(label)
        for path in files[split_at:]:
            validation_x.append(load_image(path))
            validation_y.append(label)

    return (
        np.asarray(train_x, dtype=np.float32),
        np.asarray(train_y, dtype=np.int64),
        np.asarray(validation_x, dtype=np.float32),
        np.asarray(validation_y, dtype=np.int64),
    )


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for truth, predicted in zip(y_true, y_pred, strict=True):
        matrix[int(truth), int(predicted)] += 1
    return matrix.tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model", type=Path, default=Path("model/keras_model.h5")
    )
    parser.add_argument(
        "--samples", type=Path, default=Path("training_samples")
    )
    parser.add_argument(
        "--output-model", type=Path, default=Path("model/cat_dog_model.h5")
    )
    parser.add_argument(
        "--metrics", type=Path, default=Path("evaluation_results.json")
    )
    parser.add_argument("--validation-count", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(42)
    keras.utils.set_random_seed(42)

    train_x, train_y, validation_x, validation_y = load_split(
        args.samples, args.validation_count
    )

    model = keras.models.load_model(args.base_model, compile=False)
    model.layers[0].trainable = False  # Keep MobileNet feature extractor frozen.
    model.layers[1].trainable = True   # Train only the small classification head.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
    ]
    history = model.fit(
        train_x,
        train_y,
        validation_data=(validation_x, validation_y),
        epochs=args.epochs,
        batch_size=16,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    validation_loss, validation_accuracy = model.evaluate(
        validation_x, validation_y, verbose=0
    )
    probabilities = model.predict(validation_x, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output_model, include_optimizer=False)

    metrics = {
        "classes": CLASS_NAMES,
        "training_samples": int(len(train_x)),
        "validation_samples": int(len(validation_x)),
        "epochs_completed": int(len(history.history["loss"])),
        "validation_loss": round(float(validation_loss), 6),
        "validation_accuracy": round(float(validation_accuracy), 6),
        "confusion_matrix": confusion_matrix(validation_y, predictions),
        "matrix_orientation": "rows=actual, columns=predicted",
    }
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print("Training completed")
    print(f"Saved model: {args.output_model}")
    print(f"Validation accuracy: {validation_accuracy:.2%}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()
