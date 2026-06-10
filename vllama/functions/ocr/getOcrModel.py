from paddleocr import PaddleOCR

def getOcrModel(language: str):
    
    ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang=language,
        )

    return ocr