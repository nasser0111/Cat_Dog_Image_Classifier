"""Generate deterministic augmented samples from the two supplied images.

These samples are intended for training in Teachable Machine when only one
source image is available per class. The original images are never modified.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps


def augment(image: Image.Image, rng: random.Random, size: int = 224) -> Image.Image:
    """Return one randomly transformed square RGB training image."""
    image = image.convert("RGB")
    base = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)

    zoom = rng.uniform(1.00, 1.18)
    scaled_size = max(size, round(size * zoom))
    transformed = base.resize((scaled_size, scaled_size), Image.Resampling.LANCZOS)

    max_offset = scaled_size - size
    left = rng.randint(0, max_offset) if max_offset else 0
    top = rng.randint(0, max_offset) if max_offset else 0
    transformed = transformed.crop((left, top, left + size, top + size))

    transformed = transformed.rotate(
        rng.uniform(-12.0, 12.0),
        resample=Image.Resampling.BICUBIC,
        fillcolor=(255, 255, 255),
    )
    if rng.random() < 0.5:
        transformed = ImageOps.mirror(transformed)

    transformed = ImageEnhance.Brightness(transformed).enhance(rng.uniform(0.82, 1.18))
    transformed = ImageEnhance.Contrast(transformed).enhance(rng.uniform(0.88, 1.15))
    transformed = ImageEnhance.Color(transformed).enhance(rng.uniform(0.88, 1.12))
    return transformed


def generate(source: Path, output_dir: Path, count: int, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    with Image.open(source) as image:
        for index in range(count):
            sample = augment(image, rng)
            sample.save(output_dir / f"sample_{index + 1:03d}.jpg", quality=92)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cat", type=Path, default=Path("images/cat.jpg"))
    parser.add_argument("--dog", type=Path, default=Path("images/dog.jpg"))
    parser.add_argument("--output", type=Path, default=Path("training_samples"))
    parser.add_argument("--count", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(args.cat, args.output / "Cat", args.count, seed=20260815)
    generate(args.dog, args.output / "Dog", args.count, seed=20260816)
    print(f"Generated {args.count} Cat and {args.count} Dog samples in {args.output}")


if __name__ == "__main__":
    main()
