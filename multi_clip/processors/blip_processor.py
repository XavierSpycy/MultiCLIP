from PIL import Image
from transformers import AutoProcessor

from ..models.config import CACHE_DIR

class BlipProcessor:
    def __init__(self, pretrained_model_name_or_path="Salesforce/blip-itm-large-coco"):
        self.processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path, 
            cache_dir=CACHE_DIR)
    
    def process(self, image: Image, text: str):
        text_in_context = self.__get_contexted_text(text)

        inputs = self.processor(
            images=image, 
            text=text_in_context, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=50, 
            truncation=True)
        
        return inputs
    
    def __call__(self, image: Image, text: str):
        return self.process(image, text)
    
    def __get_contexted_text(self, text: str):
        if text.isupper():
            text = text.lower()
        elif text[0].isupper():
            text = text[0].lower() + text[1:]
        
        return "a picture of " + text