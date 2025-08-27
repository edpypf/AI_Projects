from PIL import Image
import pytesseract #pip install pytesseract first

# Load an image using Pillow (PIL)
image = Image.open('test.png')

# Perform OCR on the image
text = pytesseract.image_to_string(image,lang='chi_sim')

print(text)