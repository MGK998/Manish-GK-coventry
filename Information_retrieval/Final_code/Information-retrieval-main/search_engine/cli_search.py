import argparse
from .storage import load_json
from .search import search

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="Your query")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--index", default="data/index.json")
    ap.add_argument("--stem", action="store_true", help="Use simple stemming")
    args = ap.parse_args()

    payload = load_json(args.index)
    if not payload:
        print("Index not found. Run the crawler first to build data/index.json")
        return

    results = search(args.q, payload, top_k=args.top, use_stemming=args.stem)
    if not results:
        print("No results.")
        return

    for i, r in enumerate(results, 1):
        print(f"{i}. {r.get('title','(no title)')} ({r.get('year','')}) [score={r.get('score')}]")
        print(f"   Publication: {r.get('publication_url')}")
        if r.get('authors'):
            print(f"   Authors: {', '.join(r.get('authors', []))}")
        print()

if __name__ == "__main__":
    main()
