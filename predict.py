"""Load the trained Keras model and classify one or more input images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:
    import tf_keras as keras
except ImportError as exc:  # pragma: no cover - friendly setup error
    raise SystemExit(
        "Missing dependency. Run: pip install -r requirements.txt"
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = (224, 224)


def read_labels(path: Path) -> list[str]:
    labels: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        _, separator, name = line.partition(" ")
        labels.append(name.strip() if separator else line.strip())
    if not labels:
        raise ValueError(f"No class labels found in {path}")
    return labels


def preprocess_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        fitted = ImageOps.fit(
            image.convert("RGB"), IMAGE_SIZE, method=Image.Resampling.LANCZOS
        )
    image_array = np.asarray(fitted, dtype=np.float32)
    normalized = (image_array / 127.5) - 1.0
    return np.expand_dims(normalized, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+", help="Input image path(s)")
    parser.add_argument(
        "--model",
        type=Path,
        default=BASE_DIR / "model" / "cat_dog_model.h5",
        help="Keras .h5 model path",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=BASE_DIR / "model" / "labels.txt",
        help="Class labels text file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = read_labels(args.labels)
    model = keras.models.load_model(args.model, compile=False)

    print("Cat & Dog Image Classifier")
    print(f"Model: {args.model.name}")
    print("-" * 45)

    for image_path in args.images:
        probabilities = model.predict(preprocess_image(image_path), verbose=0)[0]
        index = int(np.argmax(probabilities))

        print(f"Image: {image_path.name}")
        print(f"Prediction: {labels[index]}")
        print(f"Confidence: {probabilities[index]:.2%}")
        probability_text = ", ".join(
            f"{name}={score:.2%}" for name, score in zip(labels, probabilities, strict=True)
        )
        print(f"Probabilities: {probability_text}")
        print("-" * 45)


if __name__ == "__main__":
    main()
