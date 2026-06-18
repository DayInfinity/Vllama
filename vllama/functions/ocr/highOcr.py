import os
import cv2
import numpy as np

from paddleocr import PaddleOCR
from langdetect import detect
from deep_translator import GoogleTranslator
from PIL import Image, ImageDraw, ImageFont
import requests

from vllama.functions.ocr import getOcrModel


def translate_text(text, max_retries=3):
    """
    Translate text with retry mechanism and fallback.
    Returns original text if translation fails.
    """
    if not text or text.strip() == "":
        return text
    
    for attempt in range(max_retries):
        try:
            translated = GoogleTranslator(
                source="auto",
                target="en"
            ).translate(text)
            return translated
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            else:
                # Fallback: return original text with warning
                print(f"⚠️  Translation failed for '{text[:50]}...': {str(e)}")
                return text


def process_image(image_path, language: str = "en", annotation: bool = False, output_dir: str = "outputs"):

    os.makedirs(output_dir, exist_ok=True)

    image_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    translated_text_file = f"{output_dir}/{image_name}_translated.txt"
    output_image_file = f"{output_dir}/{image_name}_annotated.jpg"
    full_text_file = f"{output_dir}/{image_name}_full_text.txt"

    ocr = getOcrModel.getOcrModel(
        language= language,
    )

    result = ocr.predict(image_path)

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            20
        )
    except:
        font = ImageFont.load_default()

    text_output = []

    full_text = []

    print("\n========== OCR RESULTS ==========\n")

    for page in result:

        boxes = page["rec_polys"]
        texts = page["rec_texts"]
        scores = page["rec_scores"]

        for box, text, score in zip(
            boxes,
            texts,
            scores
        ):

            text = str(text).strip()

            if not text:
                continue

            print(text)
            full_text.append(text)

            if annotation:

                try:
                    lang = detect(text)
                except:
                    lang = "unknown"

                translated_text = text

                if lang != "en" and lang != "unknown":
                    translated_text = translate_text(text)

                text_output.append(
                    f"Detected Text : {text}\n"
                    f"Language      : {lang}\n"
                    f"Translation   : {translated_text}\n"
                    f"Confidence    : {score}\n"
                    f"{'-'*50}\n"
                )

                box = np.array(box).astype(int)

                points = [(int(x), int(y)) for x, y in box]

                draw.polygon(
                    points,
                    outline="red",
                    width=3
                )

                x = min(p[0] for p in points)
                y = min(p[1] for p in points)

                label = (
                    translated_text
                    if lang != "en"
                    else text
                )

                draw.text(
                    (x, max(0, y - 25)),
                    label,
                    fill="blue",
                    font=font
                )

    if annotation:
        with open(
            translated_text_file,
            "w",
            encoding="utf-8"
        ) as f:

            f.write("\n".join(text_output))

        image.save(output_image_file)

        print("\n========== FILES SAVED ==========")
        print(f"Translated Text File  : {translated_text_file}")
        print(f"Image File : {output_image_file}")

    with open(
        full_text_file,
        "w",
        encoding="utf-8"
    ) as f:
        f.write("\n".join(full_text))

    print(f"Full Text File : {full_text_file}")

    return full_text

# Example
# process_image("dist/testocrtel.png", language= "te", annotation= True, output_dir= "outputs")