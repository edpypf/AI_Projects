import os
from pdf2image import convert_from_path
import pytesseract
from tqdm import tqdm

# 設定資料夾路徑
pdf_folder = "."         # 原始 PDF 位置
output_folder = "./pdf_ocr"   # 輸出 TXT 位置

os.makedirs(output_folder, exist_ok=True)

# 批次處理所有 PDF 檔案
for filename in tqdm(os.listdir(pdf_folder), desc="🔍 OCR PDFs"):
    if not filename.endswith(".pdf"):
        continue

    pdf_path = os.path.join(pdf_folder, filename)
    txt_path = os.path.join(output_folder, filename.replace(".pdf", ".txt"))

    try:
        # 將 PDF 每一頁轉成圖片（DPI 建議 300）
        images = convert_from_path(pdf_path, dpi=300)

        full_text = ""
        for img in images:
            # OCR 擷取文字，保留格式
            text = pytesseract.image_to_string(img, config='--psm 1')  # 保留 layout
            full_text += text + "\n" + "="*50 + "\n"  # 分隔每頁

        # 儲存為文字檔
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)

    except Exception as e:
        print(f"❌ Failed to process {filename}: {e}")
