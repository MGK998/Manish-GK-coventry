
import requests
from search_engine.parser import parse_publication_page

URL = "https://pureportal.coventry.ac.uk/en/publications/gradient-coil-design-for-mri-by-neural-networks/"

def main():
    print(f"Debugging {URL}...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    
    try:
        resp = requests.get(URL, headers=headers, timeout=30)
        print(f"Status Code: {resp.status_code}")
        resp.raise_for_status()
        
        html = resp.text
        print(f"HTML Length: {len(html)}")
        
        pub = parse_publication_page(URL, html)
        print(f"Parsed Dict Keys: {pub.keys()}")
        print(f"Title: {pub.get('title')}")
        print(f"Year: {pub.get('year')}")
        print(f"Authors: {pub.get('authors')}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
