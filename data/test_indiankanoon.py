import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig

async def main():
    urls = [
        "https://indiankanoon.org/doc/1564220/",
        "https://indiankanoon.org/doc/1683365/"
    ]
    browser_cfg = BrowserConfig(headless=True, verbose=False)
    run_cfg = CrawlerRunConfig(cache_mode="bypass", page_timeout=45000, wait_until="networkidle")
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        for url in urls:
            print(f"\n=== Crawling {url} ===")
            result = await crawler.arun(url, config=run_cfg)
            print(f"Success: {result.success}")
            if result.success and result.markdown:
                print(f"Markdown len: {len(result.markdown)}")
                print(result.markdown[:3000])
                print(f"Links found: {len(result.links.get('internal', [])) if result.links else 0} internal")
            else:
                print(f"Failed: {result.error_message[:800] if result.error_message else 'unknown'}")
                if result.html:
                    print(f"HTML len: {len(result.html)}")

asyncio.run(main())
