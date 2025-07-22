import os
import json
import asyncio
import hashlib
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.tools.playwright.base import PlaywrightToolSpec

PDF_PATH = "data_pdf"
URL_FILE = "data_web/urls.txt"
PROCESSED_LOG = "data/processed_log.json"

# ------------------ Helpers for tracking ------------------

def load_processed_log():
    if os.path.exists(PROCESSED_LOG):
        with open(PROCESSED_LOG, "r") as f:
            return json.load(f)
    return {"pdfs": [], "urls": []}

def save_processed_log(log):
    os.makedirs(os.path.dirname(PROCESSED_LOG), exist_ok=True)
    with open(PROCESSED_LOG, "w") as f:
        json.dump(log, f, indent=2)

def hash_file(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# ------------------ Load PDFs with Skipping ------------------

def load_pdfs():
    if not os.path.exists(PDF_PATH):
        print("📂 No PDF folder found.")
        return []

    print("📄 Loading PDF documents...")
    processed = load_processed_log()
    already_done_hashes = processed["pdfs"]
    new_docs = []

    for filename in os.listdir(PDF_PATH):
        if not filename.lower().endswith(".pdf"):
            continue

        filepath = os.path.join(PDF_PATH, filename)
        file_hash = hash_file(filepath)
        if file_hash in already_done_hashes:
            print(f"⏩ Skipping already processed PDF: {filename}")
            continue

        print(f"📥 Reading new PDF: {filename}")
        reader = SimpleDirectoryReader(input_files=[filepath])
        docs = reader.load_data()

        # Add source metadata for traceability
        for doc in docs:
            doc.metadata.update({
                "source": filepath,
                "file_name": filename,
                "file_path": filepath,
                "file_type": "application/pdf"
            })

        new_docs.extend(docs)
        already_done_hashes.append(file_hash)

    processed["pdfs"] = already_done_hashes
    save_processed_log(processed)
    return new_docs


# ------------------ Load URLs with Skipping ------------------

async def load_urls_async():
    if not os.path.exists(URL_FILE):
        print("🌐 No URL file found.")
        return []

    with open(URL_FILE, 'r') as f:
        urls = [line.strip() for line in f if line.strip().startswith("http")]

    if not urls:
        print("⚠️ No valid URLs found.")
        return []

    processed = load_processed_log()
    already_scraped_urls = processed["urls"]

    tool_spec = PlaywrightToolSpec()
    browser = await tool_spec.create_async_playwright_browser()

    documents = []
    for url in urls:
        if url in already_scraped_urls:
            print(f"⏩ Skipping already scraped URL: {url}")
            continue

        try:
            print(f"🌍 Scraping: {url}")
            page = await browser.new_page()
            await page.goto(url)
            await page.wait_for_load_state("networkidle")
            text = await page.locator("body").inner_text()
            documents.append(Document(
                text=text,
                metadata={
                    "source": url,
                    "file_name": f"{url.replace('https://', '').replace('/', '_')}.html",
                    "file_type": "text/html"
                }
            ))
            await page.close()
            already_scraped_urls.append(url)
        except Exception as e:
            print(f"❌ Failed to scrape {url}: {e}")

    await browser.close()

    processed["urls"] = already_scraped_urls
    save_processed_log(processed)

    print(f"✅ Scraped {len(documents)} new URLs.")
    return documents

# ------------------ Main Document Loader ------------------

def load_docs():
    pdf_docs = load_pdfs()
    try:
        url_docs = asyncio.run(load_urls_async())
    except RuntimeError:
        import nest_asyncio
        nest_asyncio.apply()
        url_docs = asyncio.run(load_urls_async())

    print(f"📦 Loaded {len(pdf_docs)} PDF docs and {len(url_docs)} URL docs.")
    return pdf_docs + url_docs


