from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    device_map = device,
    torch_dtype = torch.float16
)

lora_config = LoraConfig (
    
    task_type = TaskType.CAUSAL_LM, 
    target_modules=['q_proj', "k_proj", "v_proj"]
)
model = get_peft_model(model, lora_config)

from transformers import TrainingArguments, Trainer


train_args = TrainingArguments(
    num_train_epochs = 10, # we will go throught the dataset from start to finish 10 times
    learning_rate=0.001, 
    logging_steps = 25, # we want to see the result in every 25 steps it runs 
    fp16 = False # float point set to 16 to speed it up, set to "True" if you are on GPU
)

trainer = Trainer(
    args = train_args,
    model = model, 
    train_dataset=data["train"]
)


trainer.train()

# save the model
trainer.save_model("./my-qwen")
tokenizer.save_pretrained("./my-qwen")

ask_llm = pipeline(
  task="text-generation",
  model="./my-qwen",
  tokenizer='./my-qwen',
  device=device,
  torch_dtype=dtype
)

print(ask_llm("Who is Scott Lai?")[0]["generated_text"])