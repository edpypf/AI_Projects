import requests, json, time
from bs4 import BeautifulSoup
from tqdm import tqdm
import trafilatura
import pytesseract
from PIL import Image
from io import BytesIO

BASE_URL = "https://arxiv.org"

def get_recent_papers(category="cs.CL", max_count=200):
    url = f"{BASE_URL}/list/{category}/recent"
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    entries = soup.select("dt a[href^='/abs/']")
    paper_links = list({BASE_URL + a['href'] for a in entries})[:max_count]
    print(paper_links)
    return paper_links

def parse_abs_page(url):
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")

    # 基本資訊
    title = soup.find("h1", class_="title").text.replace("Title:", "").strip()
    authors = soup.find("div", class_="authors").text.replace("Authors:", "").strip()
    date_line = soup.find("div", class_="dateline")
    date = date_line.text.strip() if date_line else ""

    # 嘗試抽取文字摘要
    abstract_div = soup.find("blockquote", class_="abstract")
    if abstract_div:
        raw_html = str(abstract_div)
        cleaned = trafilatura.extract(raw_html)
        abstract_text = cleaned.replace("Abstract:","").strip() if cleaned else ""
    else:
        abstract_text = ""

    # 如果無摘要文字，使用 OCR 偵測圖片
    if not abstract_text:
        images = soup.find_all("img")
        for img in images:
            img_url = BASE_URL + img['src']
            image_resp = requests.get(img_url)
            img_pil = Image.open(BytesIO(image_resp.content))
            abstract_text = pytesseract.image_to_string(img_pil).strip()
            if abstract_text:
                break

    return {
        "url": url,
        "title": title,
        "authors": authors,
        "date": date,
        "abstract": abstract_text
    }

def scrape_arxiv(category="cs.CL", max_count=200, output_file="arxiv_clean.json"):
    paper_urls = get_recent_papers(category, max_count)
    results = []
    for url in tqdm(paper_urls, desc="Scraping arXiv"):
        try:
            paper_data = parse_abs_page(url)
            results.append(paper_data)
            time.sleep(0.5)  # 避免被封鎖
        except Exception as e:
            print(f"Error parsing {url}: {e}")
            continue

    # 儲存 JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 完成，共儲存 {len(results)} 篇論文摘要於 {output_file}")

# 範例執行
if __name__ == "__main__":
    scrape_arxiv("cs.CL")
