
import json
from collections import Counter, defaultdict

PATH = 'c:/Users/Dell/Desktop/Information-retrieval-main/data/publications.jsonl'

def main():
    with open(PATH, 'r', encoding='utf-8') as f:
        pubs = [json.loads(line) for line in f]
    
    years = [p.get('year', 'Unknown') for p in pubs]
    counts = Counter(years)
    
    print(f"Total Publications: {len(pubs)}")
    print("Yearly Counts:")
    for y in sorted(counts.keys()):
        print(f"{y}: {counts[y]}")
        
    # Check 2010-2025
    missing = []
    print("\nGap Analysis (Target 10):")
    for y in range(2010, 2026):
        c = counts.get(str(y), 0)
        if c < 10:
            missing.append(f"{y} ({c})")
            
    if missing:
        print(f"MISSING: {', '.join(missing)}")
    else:
        print("SUCCESS! All years 2010-2025 have 10+ papers.")

if __name__ == "__main__":
    main()
