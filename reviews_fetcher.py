import os, re, requests
from bs4 import BeautifulSoup

ASIN_RE = re.compile(r"(?:/dp/|/gp/product/|/product-reviews/)([A-Z0-9]{10})")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "en-IN,en;q=0.9",
}

def extract_asin(url: str):
    """Amazon link se ASIN nikalta hai (amzn.in short links bhi chalte hain)."""
    url = url.strip()
    if "amzn." in url or "a.co" in url:          # short link -> expand karo
        try:
            url = requests.head(url, allow_redirects=True, headers=HEADERS, timeout=10).url
        except requests.RequestException:
            return None, None
    m = ASIN_RE.search(url)
    domain = re.search(r"amazon\.[a-z.]+", url)
    return (m.group(1) if m else None), (domain.group(0) if domain else "amazon.in")


def fetch_via_api(asin, domain, max_pages=5):
    """Rainforest API (key env variable RAINFOREST_API_KEY mein rakho)."""
    key = os.getenv("RAINFOREST_API_KEY")
    if not key:
        return []
    reviews = []
    for page in range(1, max_pages + 1):
        r = requests.get("https://api.rainforestapi.com/request", params={
            "api_key": key, "type": "reviews", "amazon_domain": domain,
            "asin": asin, "page": page}, timeout=30)
        if r.status_code != 200:
            break
        items = r.json().get("reviews", [])
        if not items:
            break
        for it in items:
            reviews.append({"text": f"{it.get('title','')}. {it.get('body','')}",
                            "rating": it.get("rating")})
    return reviews


def fetch_via_scraping(asin, domain):
    """Best-effort: product page par jo top reviews dikhte hain wahi milenge."""
    try:
        r = requests.get(f"https://www.{domain}/dp/{asin}", headers=HEADERS, timeout=15)
        if r.status_code != 200 or "captcha" in r.text.lower():
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        out = []
        for div in soup.select('[data-hook="review"]'):
            body = div.select_one('[data-hook="review-body"]')
            star = div.select_one('[data-hook="review-star-rating"], [data-hook="cmps-review-star-rating"]')
            if not body:
                continue
            rating = None
            if star:
                m = re.search(r"([\d.]+) out of", star.get_text())
                rating = float(m.group(1)) if m else None
            out.append({"text": body.get_text(" ", strip=True), "rating": rating})
        return out
    except requests.RequestException:
        return []


def get_reviews(url, dataset_lookup=None):
    """Order: dataset -> API -> scraping. Returns (reviews, source, asin)."""
    asin, domain = extract_asin(url)
    if not asin:
        return [], "invalid_link", None
    if dataset_lookup:
        rows = dataset_lookup(asin)          # aapka existing dataset function
        if rows:
            return rows, "dataset", asin
    rows = fetch_via_api(asin, domain)
    if rows:
        return rows, "api", asin
    rows = fetch_via_scraping(asin, domain)
    return rows, ("scraping" if rows else "not_found"), asin