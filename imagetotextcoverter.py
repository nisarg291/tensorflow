import pyautogui
# you need to dwn pytesseract from https://github.com/UB-Mannheim/tesseract/wiki and then install it.
from PIL import Image
from pytesseract import *
pytesseract.tesseract_cmd=r'';
img=Image.open('demo.ong')
