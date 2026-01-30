
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

url = "https://pureportal.coventry.ac.uk/en/organisations/ics-research-centre-for-computational-science-and-mathematical-mo/publications/"
try:
    resp = requests.get(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
        }, timeout=10)
    print(f"Status Code: {resp.status_code}")
    print(f"Content Length: {len(resp.text)}")
    
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.select("a"):    # trying generic 'a' tag selector first
        href = a.get("href")
        if not href:
            continue
        full = urljoin(url, href)
        links.append(full)
    
    print(f"Total links: {len(links)}")
    # Print pagination like links
    for l in links:
        if "?page=" in l:
            print(f"Pagination link: {l}")
            
except Exception as e:
    print(e)
