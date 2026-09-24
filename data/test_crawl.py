import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig

async def main():
    url = "https://en.wikipedia.org/wiki/Supreme_Court_of_India"
    browser_cfg = BrowserConfig(headless=True, verbose=False)
    run_cfg = CrawlerRunConfig(cache_mode="bypass", page_timeout=30000)
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url, config=run_cfg)
        print(f"Success: {result.success}")
        print(f"Markdown len: {len(result.markdown) if result.markdown else 0}")
        if result.markdown:
            print(result.markdown[:2000])
        else:
            print("No markdown")
            print(result.error_message[:500] if result.error_message else "no error")

asyncio.run(main())
