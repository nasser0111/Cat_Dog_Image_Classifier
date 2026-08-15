"""Render the captured prediction console output as a submission screenshot."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_TEXT = BASE_DIR / "prediction_output.txt"
OUTPUT_IMAGE = BASE_DIR / "screenshots" / "prediction_output.png"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def main() -> None:
    lines = OUTPUT_TEXT.read_text(encoding="utf-8").splitlines()
    font = ImageFont.truetype(FONT_PATH, 22)
    title_font = ImageFont.truetype(FONT_PATH, 18)
    line_height = 34
    width = 1400
    height = 92 + (len(lines) * line_height) + 50

    image = Image.new("RGB", (width, height), "#0d1117")
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=18, outline="#30363d", width=2)
    draw.rectangle((0, 0, width - 1, 56), fill="#161b22")
    for x, color in [(28, "#ff5f56"), (58, "#ffbd2e"), (88, "#27c93f")]:
        draw.ellipse((x - 8, 20, x + 8, 36), fill=color)
    draw.text((120, 17), "python3 predict.py images/cat.jpg images/dog.jpg", font=title_font, fill="#8b949e")

    y = 78
    for line in lines:
        color = "#7ee787" if line.startswith(("Prediction:", "Confidence:")) else "#e6edf3"
        draw.text((36, y), line, font=font, fill=color)
        y += line_height

    OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT_IMAGE)
    print(f"Saved screenshot: {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()
