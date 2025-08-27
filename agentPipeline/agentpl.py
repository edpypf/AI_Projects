import os
import asyncio
from notion_client import Client as NotionClient
from github import Github
from pyppeteer import launch
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()

# === 設定 API 金鑰與初始化物件 ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
NOTION_TOKEN = os.getenv("NOTION_TOKEN")

notion = NotionClient(auth=NOTION_TOKEN)
github = Github(GITHUB_TOKEN)
llm = ChatOpenAI(temperature=0.3, openai_api_key=OPENAI_API_KEY)


# === 1. Brave Search（模擬版，請用真 API 替換） ===
def brave_search_sim(query):
    print(f"🔍 Brave Search 查詢：{query}")
    return [
        {"title": "LangChain GitHub", "url": "https://github.com/langchain-ai/langchain"},
        {"title": "Puppeteer Docs", "url": "https://pptr.dev/"},
    ]


# === 2. GitHub Repo 抓取 README ===
def fetch_github_readme(repo_url):
    print(f"🧑‍💻 下載 GitHub: {repo_url}")
    repo_path = "/".join(repo_url.split("/")[-2:])
    repo = github.get_repo(repo_path)
    return repo.get_readme().decoded_content.decode()


# === 3. Puppeteer 擷取網頁 HTML 內容 ===
async def fetch_html(url):
    print(f"🌐 Puppeteer 擷取：{url}")
    browser = await launch(headless=True, args=['--no-sandbox'])
    page = await browser.newPage()
    await page.goto(url)
    content = await page.content()
    await browser.close()
    return content


# === 4. Filesystem 存取 ===
def save_to_file(filename, content):
    print(f"💾 存檔：{filename}")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)


# === 5. Sequential Thinking（LLM摘要） ===
def summarize_text(text):
    print("🧠 LLM 摘要中...")
    messages = [
        SystemMessage(content="你是一個技術分析助手，幫我摘要技術內容。"),
        HumanMessage(content=f"請幫我摘要以下內容：\n{text[:3000]}"),
    ]
    result = llm(messages)
    return result.content


# === 6. Notion 發佈 ===
def post_to_notion(title, summary):
    print("📘 上傳到 Notion...")
    notion.pages.create(
        parent={"database_id": os.getenv("NOTION_DATABASE_ID")},
        properties={
            "Name": {"title": [{"text": {"content": title}}]}
        },
        children=[{
            "object": "block",
            "type": "paragraph",
            "paragraph": {"text": [{"type": "text", "text": {"content": summary}}]}
        }]
    )


# === ⛓️ Main Orchestration Pipeline ===
async def main_pipeline(user_query):
    # 1. Brave Search
    results = brave_search_sim(user_query)

    # 2. GitHub 抓取
    github_content = fetch_github_readme(results[0]["url"])
    save_to_file("github_readme.md", github_content)

    # 3. Puppeteer 擷取網頁
    html = await fetch_html(results[1]["url"])
    save_to_file("webpage.html", html)

    # 4. 摘要
    summary_input = github_content + "\n\n" + html
    summary = summarize_text(summary_input)
    save_to_file("summary.txt", summary)

    # 5. Notion
    post_to_notion(title="Agent Output: " + user_query, summary=summary)

# === 執行入口點 ===
if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(
        main_pipeline("LangChain + Puppeteer 整合學習")
    )
