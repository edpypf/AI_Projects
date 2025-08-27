from PIL import Image
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.foundation import FoundationPredictor

image = Image.open('Image1.png').convert('L')
task = 'ocr_with_boxes'  # or another supported task

detection_predictor = DetectionPredictor()
foundation_predictor = FoundationPredictor()
recognition_predictor = RecognitionPredictor(foundation_predictor)

# Pass the task as a list if you want to process multiple images
predections = recognition_predictor([image], [task], detection_predictor)

for page in predections:
    for line in page.text_lines:
        print(f"Text: {line.text}")

