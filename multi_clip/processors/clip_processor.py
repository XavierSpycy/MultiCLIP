from typing import Tuple, Dict

from PIL import Image

import spacy
from transformers import (
    CLIPImageProcessor,
    CLIPTokenizerFast
    )

from ..models.config import CACHE_DIR

class ClipProcessor:
    def __init__(self, 
                 pretrained_model_name_or_path="openai/clip-vit-large-patch14"):
        
        self.image_processor = CLIPImageProcessor.from_pretrained(
            pretrained_model_name_or_path, 
            cache_dir=CACHE_DIR)
    
        self.text_tokenizer = CLIPTokenizerFast.from_pretrained(
            pretrained_model_name_or_path,
            padding=True,
            truncation=True,
            cache_dir=CACHE_DIR)
        
        self.nlp = spacy.load("en_core_web_sm")

    def process(self, image: Image, text: str) -> Tuple[Dict, Dict]:
        image_data = self.image_processor(image, return_tensors="pt")

        text_in_context = f'The photo contains {", ".join(self.__extract_noun_phrases(text))}.'

        text_data = self.text_tokenizer(
            text_in_context, 
            max_length=77, 
            padding='max_length', 
            truncation=True, 
            return_tensors="pt")
        
        return image_data, text_data

    def __call__(self, image: Image, text: str) -> Tuple[Dict, Dict]:
        return self.process(image, text)

    def __extract_noun_phrases(self, text: str):
        doc = self.nlp(text)
        noun_phrases = []
        for chunk in doc.noun_chunks:
            noun_phrases.append(chunk.text.rstrip(',').lower())
        return noun_phrases 
