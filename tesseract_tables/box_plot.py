import re

import pytesseract
import cv2

def plot_around_char(image):
    """Plot boxes around text for each character"""
    h, w, c = image.shape
    boxes = pytesseract.image_to_boxes(image)
    for b in boxes.splitlines():
        b = b.split(' ')
        image = cv2.rectangle(image, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

    cv2.imshow('char_boxes', image)
    cv2.waitKey(0)

def plot_around_words(image, user_pattern=r'[^]*', lang='rus+eng'):
    """
    Word bounding boxes
    if there is no patter then use patter for all characters"""
    image_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang=lang)
    n_boxes = len(image_data['text'])
    for i in range(n_boxes):
        if int(image_data['conf'][i]) > 60:
            if re.match(user_pattern, image_data['text'][i]):
                (x, y, w, h) = (image_data['left'][i], image_data['top'][i], image_data['width'][i], image_data['height'][i])
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('image', image)
    cv2.waitKey(0)


