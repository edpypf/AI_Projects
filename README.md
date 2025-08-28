# 🤖 AI Agents

An extensible framework for building task-driven AI agents powered by LLMs, web search, browser automation, and memory systems.

---

## 🚀 Features

- 🌐 Web-integrated search (Brave, Google)
- 🧠 Memory and long-term context using vector stores
- 🧾 FileSystem access and structured thinking
- 🔍 GitHub search and code analysis
- 🕹 Puppeteer-based browser automation
- 📝 Notion-style task logging and summarization

---

## 🛠 Tech Stack

- Python 3.10+
- LangChain / Ollama / Llama3
- Gradio UI for interaction
- Puppeteer (via Node.js)
- Notion SDK (optional)
- GitHub Search API / Brave Search API

---

## 🔄 Workflow Diagram

```mermaid
graph TD
    Start([Start Task]) --> Search[Brave Search]
    Search --> CodeSearch[GitHub Code Lookup]
    CodeSearch --> Automation[Puppeteer Automation]
    Automation --> FileIO[Filesystem Interaction]
    FileIO --> Reasoning[Sequential Reasoning]
    Reasoning --> Log[Notion / Local Report]
    Log --> Done([Task Completed])

# AI_Projects

A collection of AI-related projects, notebooks, and pipelines for exploring machine learning, NLP, automation, and more.

---

## 📁 Folder Structure

```plaintext
AI_Projects/
│
├── 1_MCP_and_Ollama.ipynb         # Notebook: MCP & Ollama integration
├── LICENSE                        # License information
├── rag.code-workspace             # VS Code Workspace for RAG
│
├── Ollama/                        # Ollama-specific models/scripts
├── Tansformer/                    # Transformer models and utilities
├── Tesseract/                     # OCR and Tesseract demos
├── VA/                            # Voice Assistant (VA) projects
├── _Gen_AI-Course-main/           # General AI course materials
├── agentPipeline/                 # Agent pipeline & automation scripts
├── asr/                           # Automatic Speech Recognition (ASR) demos
├── rag/                           # Retrieval-Augmented Generation (RAG) modules
├── tradingview-chart-mcp/         # TradingView Chart MCP integrations
└── weather/                       # Weather data and prediction models
```

> You can generate a folder tree image using tools like [draw.io](https://app.diagrams.net/) or the `tree` command, then save as `docs/folder_structure.png`.

---

## 🖼️ Visual Overview

![Folder Structure](docs/folder_structure.png)
*This image illustrates the main directory structure. Update `docs/folder_structure.png` if you add/remove folders.*

---

## ⚡ Workflow & Usage

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/edpypf/AI_Projects.git
cd AI_Projects

# Install dependencies (example for Python projects)
pip install -r requirements.txt
```
> Each sub-project may have its own requirements. Check individual folders for details.

---

### 2. Running Notebooks

- Open notebooks (e.g., `1_MCP_and_Ollama.ipynb`) in Jupyter Lab or VS Code.
- Explore demo notebooks in subfolders for specific tasks.

---

### 3. Project-Specific Workflows

- **Ollama:** Run local Ollama models from the `Ollama/` folder.
- **Tesseract:** Use scripts in `Tesseract/` for OCR tasks.
- **VA:** Voice assistant demos in `VA/`.
- **RAG:** Retrieval-Augmented Generation modules in `rag/`.
- **TradingView:** Chart integrations in `tradingview-chart-mcp/`.

Refer to each subfolder for more detailed instructions and workflow diagrams.

---

### 4. Contribution Workflow

```bash
# Fork > Clone > Create Feature Branch
git checkout -b feature/my-feature

# Make changes, then commit
git add .
git commit -m "Add new feature"

# Push and create a Pull Request
git push origin feature/my-feature
```

---

## 📝 License

This repository is licensed under the [MIT License](./LICENSE).

## 🤝 Contributing

Pull requests and suggestions are welcome! Please see individual subfolders for specific contribution guidelines.

---

*For more details on each project, see the corresponding folder and README files.*



