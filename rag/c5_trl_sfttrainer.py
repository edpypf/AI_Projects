# for CUDA mechain run following 
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "" 

from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Prepare the dataset (must have a 'text' field)
demo_dataset = Dataset.from_list([
    {"text": "Human: What is Python?\nAssistant: Python is a programming language."},
    {"text": "Human: How do I learn coding?\nAssistant: Start with basic concepts and practice regularly."}
])

# Training arguments
training_args = TrainingArguments(
    output_dir="./trl_sft_demo",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    learning_rate=5e-4,
    logging_steps=1,
    save_steps=10,
    save_total_limit=1,
    fp16=False,
    report_to=None,
)

model_name = "gpt2"
lora_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ Initialize SFTTrainer — no config, no extras
trainer = SFTTrainer(
    model=lora_model,
    args=training_args,
    train_dataset=demo_dataset
)

# ✅ Train the model
trainer.train()