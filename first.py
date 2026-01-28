import sys
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image

from transformers import (
    BitsAndBytesConfig,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    BlipForConditionalGeneration,
    BlipProcessor,
)

# Use the BLIP-2 model
MODEL_NAME = "Salesforce/blip2-opt-2.7b"

# --- NEW PROMPT: Forces detailed description without being too complex ---
PARAGRAPH_PROMPT = "Question: Describe the details of this image, including the subject's appearance, the background, and the colors. Answer:"

def load_image(image_path: str) -> Image.Image:
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(path).convert("RGB")

def load_blip(device: torch.device) -> Tuple[Blip2ForConditionalGeneration, Blip2Processor]:
    processor = Blip2Processor.from_pretrained(MODEL_NAME)
    
    # Enable CPU offload to save VRAM
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model.eval()
    return model, processor

def generate_paragraph(
    model: Blip2ForConditionalGeneration,
    processor: Blip2Processor,
    image: Image.Image,
    device: torch.device,
    max_tokens: int = 100, # Increased slightly
) -> str:

    inputs = processor(
        images=image,
        text=PARAGRAPH_PROMPT,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"].to(device, dtype=torch.float16)
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            num_beams=5,
            
            # --- THE "SWEET SPOT" SETTINGS ---
            repetition_penalty=1.5,      # Prevents "tibet tibet tibet"
            no_repeat_ngram_size=2,      # Stops it from repeating phrases
            min_length=30,               # Forces it to write at least 30 words
            length_penalty=1.0,
            do_sample=False, 
        )
        
        gen_kwargs["pixel_values"] = pixel_values
        gen_kwargs["input_ids"] = input_ids
        
        output_ids = model.generate(**gen_kwargs)

    text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    paragraph = text.strip()

    # Clean the output (remove the Question/Answer part)
    if "Answer:" in paragraph:
        paragraph = paragraph.split("Answer:")[-1].strip()
    
    return paragraph

def generate_phrases(model, processor, image, device, num_phrases=3):
    # (Existing phrase generation logic for non-paragraph mode)
    simple_model_name = "Salesforce/blip-image-captioning-large"
    simple_processor = BlipProcessor.from_pretrained(simple_model_name)
    simple_model = BlipForConditionalGeneration.from_pretrained(simple_model_name).to(device)
    simple_model.eval()

    inputs = simple_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = simple_model.generate(
            **inputs, 
            max_new_tokens=30, 
            num_beams=5, 
            num_return_sequences=min(num_phrases, 5),
            repetition_penalty=1.2
        )
    return simple_processor.batch_decode(output_ids, skip_special_tokens=True)

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Warning: Running on CPU.")

    # Parse Arguments
    image_path = ""
    paragraph_mode = False
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter image path: ").strip().strip('"')
        
    if len(sys.argv) > 2 and sys.argv[2].lower() in ['p', 'paragraph']:
        paragraph_mode = True
    elif len(sys.argv) == 2:
        # If running manually, ask user
        paragraph_mode = input("Paragraph mode? (y/n): ").lower() == 'y'

    try:
        image = load_image(image_path)
        
        if paragraph_mode:
            print(f"Loading {MODEL_NAME}...")
            model, processor = load_blip(device)
            print("Generating description...")
            desc = generate_paragraph(model, processor, image, device)
            print("\n" + "="*40)
            print("DESCRIPTION:")
            print(desc)
            print("="*40 + "\n")
        else:
            # Fallback to simple captioning
            print("Generating phrases...")
            phrases = generate_phrases(None, None, image, device)
            for p in phrases:
                print(f"- {p}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()