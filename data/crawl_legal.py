"""
Crawl Indian Legal Judgments for D3 Fine-Tuning
Target: 1000+ judgments from IndianKanoon for summarization dataset
Uses crawl4ai SDK with concurrency and schema extraction
"""
import asyncio
import json
import random
import re
from pathlib import Path
from datetime import datetime

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# Configuration
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = RAW_DIR / "indian_legal_raw.jsonl"
SEED_URLS_FILE = Path("data/seed_urls.txt")

# Sample landmark cases + Supreme Court search pages as seeds
# IndianKanoon doc IDs are sequential - we sample recent ones
LANDMARK_URLS = [
    "https://indiankanoon.org/doc/1564220/",  # Sanjeev Yadav
    "https://indiankanoon.org/doc/1683365/",  # Preeti Kaur
    "https://indiankanoon.org/doc/1934103/",  # Kesavananda Bharati
    "https://indiankanoon.org/doc/1199182/",  # Article 21
    "https://indiankanoon.org/doc/456510/",   # Maneka Gandhi
    "https://indiankanoon.org/doc/1246399/",  # Vishaka
    "https://indiankanoon.org/doc/1373219/",  # SR Bommai
    "https://indiankanoon.org/doc/445782/",   # Mohini Jain
]

def generate_seed_urls(count=100):
    """Generate IndianKanoon doc URLs by sampling doc IDs + landmark + search pages"""
    urls = set(LANDMARK_URLS)
    
    # Add Supreme Court search pagination (JS-rendered, but crawl4ai handles)
    for page in range(0, 5):
        urls.add(f"https://indiankanoon.org/search/?formInput=supreme%20court&pagenum={page}")
    
    # Sample random doc IDs in known range (1M - 20M, recent judgments)
    # IndianKanoon has ~4 crore docs, but 1M-5M range is dense with Supreme Court
    for _ in range(count - len(urls)):
        doc_id = random.randint(100000, 5000000)
        urls.add(f"https://indiankanoon.org/doc/{doc_id}/")
    
    return list(urls)

def extract_judgment_data(markdown: str, url: str) -> dict | None:
    """Extract judgment metadata from markdown"""
    if not markdown or len(markdown) < 500:
        return None
    
    # Filter out premium/ads noise
    lines = [l for l in markdown.split('\n') if l.strip() and 
             not l.strip().startswith('[') or 'High Court' in l or 'Supreme Court' in l or 'vs' in l.lower()]
    
    # Try to find case title (pattern: "X vs Y on DATE")
    title_match = re.search(r'##\s+(.+? vs .+? on \d+.+?\d{4})', markdown, re.IGNORECASE)
    if not title_match:
        title_match = re.search(r'(.+? vs .+?)\n', markdown)
    
    title = title_match.group(1).strip() if title_match else "Unknown"
    if len(title) > 200:
        title = title[:200]
    
    # Extract court
    court_match = re.search(r'(Supreme Court|High Court|District Court)[^\n]*', markdown)
    court = court_match.group(0).strip() if court_match else "Unknown Court"
    
    # Clean markdown: remove navigation, keep judgment body
    # Judgment body is usually in code blocks or after "Present :"
    body_start = markdown.find("Present :")
    if body_start == -1:
        body_start = markdown.find("```")
    if body_start != -1:
        body = markdown[body_start:body_start+8000]
    else:
        body = markdown[2000:10000]
    
    # Remove markdown links/images for cleaner text
    body = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', body)
    body = re.sub(r'!\[.*?\]\(.*?\)', '', body)
    body = body.strip()
    
    if len(body) < 300:
        return None
    
    return {
        "url": url,
        "title": title,
        "court": court,
        "text": body[:6000],  # truncate for training
        "length": len(body),
        "crawled_at": datetime.now().isoformat(),
        "source": "indiankanoon.org"
    }

async def crawl_batch(urls: list[str], max_concurrent: int = 5):
    """Crawl batch with concurrency and pruning filter"""
    browser_cfg = BrowserConfig(headless=True, verbose=False)
    
    # Use pruning filter to remove nav/ads, keep judgment content
    md_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.48, threshold_type="fixed")
    )
    run_cfg = CrawlerRunConfig(
        cache_mode="bypass",
        page_timeout=45000,
        wait_until="networkidle",
        markdown_generator=md_generator,
        verbose=False
    )
    
    results = []
    failed = 0
    
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        # Use arun_many for concurrency
        batch_results = await crawler.arun_many(urls, config=run_cfg)
        
        for url, result in zip(urls, batch_results):
            if result.success and result.markdown:
                # Prefer fit_markdown (filtered) if available
                markdown = getattr(result.markdown, 'fit_markdown', None) or result.markdown
                if isinstance(markdown, str) and len(markdown) > 500:
                    data = extract_judgment_data(markdown, url)
                    if data:
                        results.append(data)
                        print(f"[OK] {url} -> {data['title'][:50]} ({data['length']} chars)")
                    else:
                        failed += 1
                        print(f"[SKIP] {url} -> filtered (too short/no title)")
                else:
                    failed += 1
                    print(f"[SKIP] {url} -> empty markdown")
            else:
                failed += 1
                msg = result.error_message[:100] if result.error_message else 'failed'
                print(f"[FAIL] {url} -> {msg}")
    
    return results, failed

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Crawl Indian legal judgments")
    parser.add_argument("--count", type=int, default=50, help="Number of URLs to crawl (default 50 demo, use 1000 for full)")
    parser.add_argument("--concurrent", type=int, default=5, help="Concurrent crawls")
    parser.add_argument("--seed-file", type=str, default=None, help="File with URLs (one per line)")
    args = parser.parse_args()
    
    if args.seed_file and Path(args.seed_file).exists():
        urls = [l.strip() for l in open(args.seed_file) if l.strip()]
        print(f"Loaded {len(urls)} URLs from {args.seed_file}")
    else:
        urls = generate_seed_urls(args.count)
        print(f"Generated {len(urls)} seed URLs")
        # Save seed for reproducibility
        Path("data/seed_urls.txt").write_text("\n".join(urls))
    
    print(f"Crawling {len(urls)} URLs with concurrency={args.concurrent}...")
    results, failed = await crawl_batch(urls, max_concurrent=args.concurrent)
    
    print(f"\n=== Summary ===")
    print(f"Crawled: {len(urls)}, Success: {len(results)}, Failed/filtered: {failed}")
    print(f"Success rate: {len(results)/len(urls)*100:.1f}%")
    
    if results:
        # Deduplicate batch by title
        seen = set()
        deduped = []
        for r in results:
            key = r["title"].lower().strip()
            if key not in seen and len(key) > 10:
                seen.add(key)
                deduped.append(r)
        
        print(f"After dedup (batch): {len(deduped)} unique judgments")
        
        # Append to existing raw file (don't overwrite) — merge and dedupe globally
        existing = []
        if OUTPUT_FILE.exists():
            try:
                for line in open(OUTPUT_FILE, encoding="utf-8"):
                    if line.strip():
                        existing.append(json.loads(line))
                print(f"Loaded {len(existing)} existing judgments from {OUTPUT_FILE}")
            except:
                existing = []
        
        # Merge and dedupe globally by title
        merged = {r["title"].lower().strip(): r for r in existing}
        for r in deduped:
            key = r["title"].lower().strip()
            if key not in merged:
                merged[key] = r
        
        final = list(merged.values())
        print(f"Total after merge: {len(final)} unique judgments (added {len(final)-len(existing)})")
        
        # Save merged to JSONL
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for r in final:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        
        print(f"Saved to {OUTPUT_FILE} ({OUTPUT_FILE.stat().st_size/1024:.1f} KB)")
        
        # Also save instruction format preview (from final)
        preview_file = Path("data/raw/sample_instruction.jsonl")
        with open(preview_file, "w", encoding="utf-8") as f:
            for r in final[:3]:
                inst = {
                    "instruction": "Summarize the Indian legal judgment in 5 bullets (facts, issues, holdings, reasoning, order).",
                    "input": r["text"][:1200],
                    "output": f"- Facts: {r['title']} before {r['court']}\n- Issues: [to be generated]\n- Holdings: [to be generated]\n- Reasoning: [to be generated]\n- Order: [to be generated]"
                }
                f.write(json.dumps(inst, ensure_ascii=False) + "\n")
        print(f"Preview instruction format: {preview_file} (ignored, local-only)")
        
        # Stats on final
        avg_len = sum(r["length"] for r in final) / len(final) if final else 0
        print(f"Avg judgment length: {avg_len:.0f} chars")
        print(f"Progress to 1000: {len(final)}/1000 ({len(final)/10:.0f}%)")
        print(f"\nNext: Run data/prepare.py to create train/val/test splits for QLoRA")
    else:
        print("No results — check URLs or try with --seed-file")

if __name__ == "__main__":
    asyncio.run(main())
