"""Download the 2025 Get It Done file from the City of San Diego portal to data/gid2025.csv."""
import os, sys, urllib.request

URL = ("https://seshat.datasd.org/get_it_done_reports/"
       "get_it_done_requests_closed_2025_datasd.csv")
OUT = "data/gid2025.csv"

os.makedirs("data", exist_ok=True)
if os.path.exists(OUT):
    print(f"{OUT} already exists"); sys.exit(0)
print("Downloading", URL)
try:
    urllib.request.urlretrieve(URL, OUT)
except Exception as e:
    sys.exit(f"Download failed ({e}). Get the CSV from https://data.sandiego.gov "
             "(Get It Done requests closed in 2025) and save it as data/gid2025.csv")
n = sum(1 for _ in open(OUT, encoding="utf-8")) - 1
print(f"Saved {OUT}: {n:,} rows (expected 378,669 for the Nov 2026 snapshot)")
