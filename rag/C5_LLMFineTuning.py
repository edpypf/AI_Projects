import torch
from transformers import pipeline

if torch.cuda.is_available():
    device = 0
elif torch.backends.mps.is_available():
    device = -1  # Use CPU for MPS, or use model.to("mps") after loading
else:
    device = -1  # CPU

# ask_llm = pipeline(task="text-generation", model="gpt2", device=device)
# print(ask_llm("Who is Vincent Yu.")[0]['generated_text'])


# load data 
from datasets import load_dataset

raw_data = load_dataset('json', data_files = "vincent.json")
raw_data

# raw_data["train"][0]


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Use EOS as pad token
def preprocess(sample):
    sample = sample['prompt']+ '\n' + sample['completion']
    print(sample)
    tokenized = tokenizer(
        sample,
        max_length = 128,
        truncation = True,
        padding = "max_length"    
    )

    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized
data = raw_data.map(preprocess)

print(data['train'])