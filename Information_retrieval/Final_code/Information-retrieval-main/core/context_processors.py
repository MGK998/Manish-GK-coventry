from datetime import datetime, timedelta
from django.conf import settings
import os

def crawl_dates(request):
    """
    Context processor to add crawl dates to the context.
    Reads the modification time of the index.json file to determine the last crawl date.
    Assumes a weekly crawl schedule to calculate the next crawl date.
    """
    index_path = settings.BASE_DIR / "data" / "index.json"
    last_crawl = None
    next_crawl = None

    if index_path.exists():
        timestamp = os.path.getmtime(index_path)
        last_crawl = datetime.fromtimestamp(timestamp)
        next_crawl = last_crawl + timedelta(days=7)

    return {
        "last_crawl": last_crawl,
        "next_crawl": next_crawl,
    }
