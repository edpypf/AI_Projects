import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import warnings
# huggingface-cli login: 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
# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# The original model name appears to be incorrect
# Let's use a valid Mistral model instead
repo_id = "mistralai/Mistral-7B-Instruct-v0.1"

try:
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(repo_id)
    
    print("Loading model...")
    # Use AutoModelForCausalLM instead of VoxtralForConditionalGeneration
    model = AutoModelForCausalLM.from_pretrained(
        repo_id, 
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device if device == "cuda" else None,
        low_cpu_mem_usage=True
    )
    
    conversation = [
        {
            "role": "user",
            "content": "Why should AI models be open-sourced?"
        }
    ]
    
    print("Processing inputs...")
    # Apply chat template and get the proper input format
    inputs = processor.apply_chat_template(
        conversation, 
        return_tensors="pt",
        add_generation_prompt=True
    )
    
    # Move inputs to device
    inputs = inputs.to(device)
    
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7,
            pad_token_id=processor.eos_token_id
        )
    
    # Decode only the new tokens (response)
    response = processor.decode(
        outputs[0][inputs.shape[1]:], 
        skip_special_tokens=True
    )
    
    print("\nGenerated response:")
    print("=" * 80)
    print(response)
    print("=" * 80)
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you have the transformers library installed:")
    print("pip install transformers torch")
    
except Exception as e:
    print(f"Error occurred: {e}")
    print(f"Error type: {type(e).__name__}")
