
import time
import subprocess
import os
from datetime import datetime

def run_weekly_crawl():
    print(f"[{datetime.now()}] Starting scheduled crawl...")
    try:
        # Run the management command
        # Using subprocess to run the django management command
        result = subprocess.run(["python", "manage.py", "run_crawl"], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Errors: {result.stderr}")
    except Exception as e:
        print(f"Scheduler failed: {e}")

def main():
    print("Crawl Scheduler started. Checking for schedule every hour...")
    # We want it to run once a week, e.g., every Sunday at 02:00 AM
    SCHEDULE_DAY = 6 # Sunday (0=Monday, 6=Sunday)
    SCHEDULE_HOUR = 2 # 2 AM
    
    last_run_date = None

    while True:
        now = datetime.now()
        
        # Check if it's Sunday at 2 AM and we haven't run it today already
        if now.weekday() == SCHEDULE_DAY and now.hour == SCHEDULE_HOUR:
            current_date = now.date()
            if last_run_date != current_date:
                run_weekly_crawl()
                last_run_date = current_date
        
        # Sleep for 30 minutes before checking again
        time.sleep(1800)

if __name__ == "__main__":
    main()
