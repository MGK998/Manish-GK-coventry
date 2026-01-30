
import json
from repair_slugs import repair_url

def main():
    path = 'c:/Users/Dell/Desktop/Information-retrieval-main/data/publications.jsonl'
    with open(path, 'r', encoding='utf-8') as f:
        line = f.readline()
        if not line:
            print("Empty file")
            return
        p = json.loads(line)
        original = p['publication_url']
        title = p['title']
        repaired = repair_url(original, title)
        
        print(f"Title: {repr(title)}")
        print(f"Original URL: {original}")
        print(f"Repaired URL: {repaired}")
        print(f"Matches? {original == repaired}")

if __name__ == "__main__":
    main()
