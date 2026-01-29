
import re

SITEMAP_FILE = "sitemap_full.xml"
TARGET_COUNT = 1000

def main():
    print(f"Reading local sitemap {SITEMAP_FILE}...")
    try:
        with open(SITEMAP_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading sitemap: {e}")
        return

    print(f"Sitemap read ({len(content)} bytes). Parsing...")
    
    # Simple regex to find locs
    urls = re.findall(r'<loc>(.*?)</loc>', content)
    total = len(urls)
    print(f"Found {total} total URLs in sitemap.")
    
    # Take the LAST 1000 (most recent)
    selected_urls = urls[-TARGET_COUNT:]
        
    print(f"Selected {len(selected_urls)} most recent URLs.")
    
    # Save to file
    with open('filtered_urls.txt', 'w', encoding='utf-8') as f:
        for url in selected_urls:
            f.write(url + '\n')
            
    # Print sample
    print("Sample (Newest):")
    for u in selected_urls[-3:]:
        print(u)

if __name__ == "__main__":
    main()
