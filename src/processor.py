import torch
from transformers import pipeline

class NLPProcessor:
    def __init__(self, model="gpt2"):
        self.pipe = pipeline("text-generation", model=model)

    def process(self, text: str):
        return self.pipe(text, max_length=50)
