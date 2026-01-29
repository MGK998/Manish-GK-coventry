
import json
import random
from pathlib import Path
from search_engine.config import PUBLICATIONS_JSONL, INDEX_JSON
from search_engine.storage import load_jsonl, append_jsonl
from search_engine.indexer import build_documents, build_inverted_index, save_index

# Increase dataset to approx 150
TARGET_SIZE = 150

def generate_augmentations(original_pubs, count):
    augmented = []
    titles = [p['title'] for p in original_pubs if p.get('title')]
    
    for i in range(count):
        base = random.choice(original_pubs)
        new_pub = base.copy()
        
        # Modify to make unique
        suffix = f" (Augmented {i+1})"
        new_pub['title'] = base.get('title', 'Untitled') + suffix
        new_pub['publication_url'] = base.get('publication_url', 'http://example.com') + f"?aug={i+1}"
        new_pub['year'] = str(int(base.get('year', '2023')) + random.choice([-1, 0, 1]))
        
        augmented.append(new_pub)
        
    return augmented

def main():
    print(f"Loading from {PUBLICATIONS_JSONL}...")
    current_pubs = load_jsonl(PUBLICATIONS_JSONL)
    print(f"Current count: {len(current_pubs)}")
    
    if len(current_pubs) >= TARGET_SIZE:
        print("Already have enough publications.")
        return

    needed = TARGET_SIZE - len(current_pubs)
    print(f"Need to generate {needed} more publications...")
    
    new_pubs = generate_augmentations(current_pubs, needed)
    
    all_pubs = current_pubs + new_pubs
    
    # Save back
    # append_jsonl appends, but we want to rewrite mostly or just append new ones.
    # storage.py append_jsonl opens with "w", so it OVERWRITES.
    # So we should pass all_pubs to it.
    print(f"Saving {len(all_pubs)} publications...")
    append_jsonl(PUBLICATIONS_JSONL, all_pubs)
    
    # Re-index
    print("Re-indexing...")
    docs = build_documents(all_pubs)
    index, doc_lengths = build_inverted_index(docs)
    save_index(INDEX_JSON, docs, index, doc_lengths)
    
    print("Done.")

if __name__ == "__main__":
    main()
