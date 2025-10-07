# LLM Data Cleansing & Agentic Engineering Cheat Sheet

---

## 1. Data Cleansing (Python + GPU)

**Key Libraries:**  
- `pandas`, `numpy`, `re` (regex), `nltk`, `spacy`, `datasets` (HuggingFace), `torch`, `multiprocessing`

**Techniques:**  
- Text normalization (lowercase, remove special chars)
- Tokenization
- Stopword removal
- Deduplication
- Filtering (length, quality, language)
- Parallel processing (multiprocessing, Dask)

**Sample Code: Efficient Text Cleaning with Pandas & Multiprocessing**
```python
import pandas as pd
import re
from multiprocessing import Pool, cpu_count

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if __name__ == "__main__":
    df = pd.read_csv('raw_texts.csv')
    with Pool(cpu_count()) as pool:
        df['clean_text'] = pool.map(clean_text, df['text'])
    df.to_csv('clean_texts.csv', index=False)
```

---

## 2. GPU Acceleration (PyTorch Example)

**Key Points:**
- Use `torch.device('cuda')` if GPU available.
- DataLoader with `num_workers` for throughput.

**Sample Training Loop**
```python
import torch
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = TensorDataset(torch.randn(1000, 10), torch.randint(0, 2, (1000,)))
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

model = torch.nn.Linear(10, 2).to(device)
optimizer = torch.optim.Adam(model.parameters())

for X, y in loader:
    X, y = X.to(device), y.to(device)
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(X), y)
    loss.backward()
    optimizer.step()
```

---

## 3. Data Cleansing & Agentic Engineering on AWS

**Key Services:**
- **S3:** Data storage (raw, processed, versioned)
- **SageMaker Processing Jobs:** Large-scale, managed data preprocessing
- **Glue:** ETL for structured/unstructured data
- **SageMaker Training Jobs:** Managed GPU training
- **SageMaker Pipelines / Step Functions:** Workflow orchestration (agentic pipelines)
- **IAM:** Secure access control

**Sample: SageMaker Processing Job (with Pandas Script)**
```python
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker import get_execution_role, Session

role = get_execution_role()
script_processor = ScriptProcessor(
    image_uri='763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.0-cpu-py310-ubuntu20.04-sagemaker',
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge'
)

script_processor.run(
    code='cleaning_script.py',
    inputs=[ProcessingInput(source='s3://my-bucket/raw_data/', destination='/opt/ml/processing/input')],
    outputs=[ProcessingOutput(source='/opt/ml/processing/output', destination='s3://my-bucket/cleaned_data/')],
)
```
*`cleaning_script.py` contains your pandas cleaning logic.*

---

## 4. Agentic Engineering Concepts

- **Agent:** Autonomous unit that takes actions (e.g., LLM + tools)
- **Workflow orchestration:** Chain data cleansing, validation, training, evaluation
- **Tools:** [LangChain](https://python.langchain.com/), [LlamaIndex](https://docs.llamaindex.ai/)

**LangChain Example:**
```python
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool

def custom_tool(input_str):
    # Example tool, could be data lookup, cleaning, etc.
    return input_str[::-1]

tools = [Tool(name="ReverseText", func=custom_tool, description="Reverse a string")]
llm = OpenAI()
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
print(agent.run("Reverse the word 'hello'"))
```

---

## 5. AWS: Typical Workflow

1. Upload raw data to **S3**
2. Run a **SageMaker Processing Job** for cleansing
3. Store outputs back in **S3**
4. Train LLM using **SageMaker Training Job** (GPU instance)
5. Deploy with **SageMaker Endpoint**
6. Orchestrate via **SageMaker Pipelines** or **Step Functions**

---

# Mock Interview Q&A

---

### Q1. **How do you handle text deduplication at scale for LLM pre-training data?**
**A:**  
I use a combination of pandas for initial cleaning and hashing for deduplication. For large datasets, I leverage Dask or PySpark to process data in parallel, either locally or on AWS EMR. On AWS, I store data in S3 and use SageMaker Processing Jobs or Glue for distributed execution.

---

### Q2. **Describe how you would accelerate data processing using GPUs.**
**A:**  
Although GPUs are mainly for model training, some libraries (like RAPIDS cuDF) allow GPU-accelerated dataframes. For typical text cleaning, CPU parallelization (multiprocessing, Dask) is more common. For LLM tokenization and batching, I use DataLoader with multi-workers to keep the GPU fed efficiently.

---

### Q3. **Have you used AWS SageMaker for data cleansing? How?**
**A:**  
Yes. I write my data cleaning logic in a Python script, upload it to S3, and launch a SageMaker Processing Job that runs the script in a managed container. This allows scalable, distributed preprocessing. I use S3 as both the input and output data store.

---

### Q4. **How would you orchestrate an LLM data pipeline with agentic engineering principles on AWS?**
**A:**  
I design each step (ingestion, cleansing, training, validation, deployment) as an agent/task. I use SageMaker Pipelines or AWS Step Functions to chain the steps, handle dependencies, and manage failures. Each step can trigger the next, and agents can interact with S3, invoke Lambda, or launch training jobs.

---

### Q5. **How do you secure sensitive data during LLM training on AWS?**
**A:**  
I use IAM roles for least-privilege access, enable S3 bucket encryption, and make sure data is always encrypted in transit and at rest. SageMaker jobs run in a VPC for network isolation, and I use CloudWatch for audit logging.

---

### Q6. **What’s the difference between running a data cleansing pipeline locally with Python vs on AWS?**
**A:**  
Locally, resources are limited by my hardware, scaling is manual, and collaboration is harder. On AWS, I get scalable, managed resources, can process much larger datasets, and have integrated security and monitoring. AWS also allows for workflow orchestration and automation.

---

### Q7. **How can you optimize costs when working with large LLM datasets on AWS?**
**A:**  
I use Spot Instances for non-critical jobs, compress intermediate data, leverage efficient data formats (Parquet), and monitor usage with AWS Budgets and Cost Explorer. I also clean up unused resources and automate shutdowns.

---

## Need more examples, deeper explanations, or a walkthrough? Let me know!