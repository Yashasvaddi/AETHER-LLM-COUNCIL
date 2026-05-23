from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
import config
import time
import random
import logging

logger = logging.getLogger(__name__)

# User agent pool for rotation
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
]

def search_web(query, max_results=None):
    """
    Search the web using DuckDuckGo.
    Returns list of search results with title, url, snippet.
    """
    if max_results is None:
        max_results = config.MAX_SEARCH_RESULTS
    
    logger.info(f"🔍 Searching: {query}")
    results = []
    try:
        with DDGS() as ddgs:
            for result in ddgs.text(query, max_results=max_results):
                results.append({
                    'title': result.get('title', ''),
                    'url': result.get('href', ''),
                    'snippet': result.get('body', '')
                })
        logger.info(f"✓ Found {len(results)} search results")
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
    
    return results

def scrape_with_requests(url, timeout):
    """Direct scraping with requests library."""
    logger.debug(f"→ Trying direct scrape: {url}")
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0',
    }
    
    response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    
    # Handle encoding properly
    if response.encoding is None:
        response.encoding = 'utf-8'
    
    logger.info(f"✓ Direct scrape successful: {url}")
    return response.text

def scrape_with_playwright(url, timeout):
    """Fallback: Use playwright for JS-rendered pages."""
    try:
        logger.debug(f"→ Trying Playwright fallback: {url}")
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                viewport={'width': 1920, 'height': 1080}
            )
            page = context.new_page()
            
            # Set shorter timeout and don't wait for everything
            page.set_default_timeout(timeout * 1000)
            page.goto(url, wait_until='commit')  # Don't wait for full load
            page.wait_for_timeout(2000)  # Wait 2 seconds for basic content
            
            content = page.content()
            browser.close()
            logger.info(f"✓ Playwright scrape successful: {url}")
            return content
    except ImportError:
        return None
    except Exception as e:
        logger.warning(f"⚠ Playwright fallback failed for {url}: {e}")
        return None

def scrape_page(url, timeout=None):
    """
    Scrape content from a URL with multiple fallback strategies.
    Returns cleaned text content or None if failed.
    """
    if timeout is None:
        timeout = config.SCRAPE_TIMEOUT
    
    html_content = None
    
    # Strategy 1: Direct requests with enhanced headers
    try:
        html_content = scrape_with_requests(url, timeout)
    except Exception as e:
        logger.debug(f"⚠ Direct scrape failed for {url}: {e}")
    
    # Strategy 2: Playwright fallback for JS-heavy or blocking sites
    if not html_content:
        try:
            html_content = scrape_with_playwright(url, timeout)
        except Exception as e:
            logger.warning(f"❌ All scrape methods failed for {url}")
    
    # Parse HTML if we got any content
    if html_content:
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script, style, nav, footer
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'noscript']):
                element.decompose()
            
            # Get text
            text = soup.get_text(separator='\n', strip=True)
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            cleaned = '\n'.join(lines)
            
            if cleaned and len(cleaned) > 100:
                logger.debug(f"✓ Extracted {len(cleaned)} chars from {url}")
                return cleaned
            else:
                logger.debug(f"⚠ Content too short from {url}")
                return None
        except Exception as e:
            logger.error(f"❌ HTML parsing error for {url}: {e}")
            return None
    
    return None

def chunk_text(text, chunk_size=500):
    """
    Chunk text into semantic pieces.
    Simple sentence-based chunking.
    """
    sentences = text.replace('! ', '!|').replace('? ', '?|').replace('. ', '.|').split('|')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        sentence_length = len(sentence.split())
        
        if current_length + sentence_length > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def collect_evidence(factor, orientation='pro'):
    """
    Collect evidence for a factor with a specific orientation (pro or con).
    Returns list of evidence chunks with metadata.
    """
    # Create search query
    if orientation == 'pro':
        query = f"{factor} benefits advantages success cases validation"
    else:
        query = f"{factor} risks failures criticisms problems disadvantages"
    
    search_results = search_web(query)
    
    evidence_chunks = []
    pages_scraped = 0
    
    for idx, result in enumerate(search_results, 1):
        if pages_scraped >= config.MAX_SCRAPED_PAGES_PER_FACTOR:
            logger.info(f"✓ Reached max pages limit ({config.MAX_SCRAPED_PAGES_PER_FACTOR})")
            break
        
        url = result['url']
        logger.info(f"📄 Scraping page {idx}/{len(search_results)}: {url[:60]}...")
        content = scrape_page(url)
        
        if content:
            chunks = chunk_text(content)
            logger.info(f"✓ Extracted {len(chunks)} chunks from page")
            for chunk in chunks[:3]:  # Max 3 chunks per page
                evidence_chunks.append({
                    'text': chunk,
                    'source': url,
                    'title': result['title'],
                    'orientation': orientation
                })
            pages_scraped += 1
            time.sleep(0.5)  # Be polite
        else:
            logger.debug(f"⚠ No content extracted from {url}")
    
    logger.info(f"✓ Collected {len(evidence_chunks)} evidence chunks ({orientation})")
    return evidence_chunks

def collect_all_evidence(factor, enable_scraping=True):
    """
    Collect both pro and con evidence for a factor.
    If enable_scraping=False, returns empty evidence.
    """
    if not enable_scraping:
        logger.info("🚫 Web scraping disabled - skipping evidence collection")
        return {
            'pro': [],
            'con': []
        }
    
    pro_evidence = collect_evidence(factor, 'pro')
    con_evidence = collect_evidence(factor, 'con')
    
    return {
        'pro': pro_evidence,
        'con': con_evidence
    }
