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
