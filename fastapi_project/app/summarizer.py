import torch
from transformers import Pipeline, pipeline

class TextSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        print(f"Loading model {model_name}...")
        # Check for GPU
        device = 0 if torch.cuda.is_available() else -1
        self.summarizer = pipeline(
            "summarization", 
            model=model_name, 
            device=device
        )
        print("Model loaded.")

    def summarize(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        chunk_size = 3000 
        
        if len(text) <= chunk_size:
            return self._summarize_chunk(text, max_length, min_length)

        # Split text into chunks
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        summaries = []
        
        print(f"Processing {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks):
            summary = self._summarize_chunk(chunk, max_length=max_length, min_length=min_length)
            summaries.append(summary)
        
        full_summary = " ".join(summaries)
        
        return full_summary

    def _summarize_chunk(self, text, max_length, min_length):
        try:
            # BART-large-CNN works best with proper parameters for abstractive summarization
            output = self.summarizer(
                text, 
                max_length=max_length, 
                min_length=min_length, 
                do_sample=False,
                truncation=True,
                max_new_tokens=max_length,  # Fix truncation warning
                num_beams=4,  # Better quality summaries
                length_penalty=2.0,  # Encourage longer, more complete summaries
                early_stopping=True
            )
            return output[0]['summary_text']
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
            return ""
