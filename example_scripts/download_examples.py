#!/usr/bin/env python
"""download_examples.py - Download 5 example SWC files from BIL.

Downloads _reg.swc files which are registered to the Allen CCF coordinate system.
"""

import json
import re
import ssl
import urllib.request
from pathlib import Path

# Configuration
OUTPUT_DIR = Path("example_data")
NUM_FILES = 5

# BIL API endpoints
API_BASE = "https://api.brainimagelibrary.org"
DOWNLOAD_BASE = "https://download.brainimagelibrary.org"
MORPHOLOGY_SUBMISSION_UUID = "0fcde5fdd6f7ccb2"


def get_ssl_context():
    """Create SSL context for BIL API."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def api_get(endpoint: str) -> dict:
    """Make a GET request to the BIL API."""
    url = f"{API_BASE}/{endpoint}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=get_ssl_context()) as response:
        return json.loads(response.read().decode())


def download_file(url: str, output_path: Path) -> bool:
    """Download a file from URL."""
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, context=get_ssl_context()) as response:
            output_path.write_bytes(response.read())
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Querying BIL API...")
    result = api_get(f"query/submission?submission_uuid={MORPHOLOGY_SUBMISSION_UUID}")
    bildids = result.get("bildids", [])[:10]  # Check first 10 entries

    swc_urls = []
    for bildid in bildids:
        result = api_get(f"retrieve?bildid={bildid}")
        for entry in result.get("retjson", []):
            dataset = entry.get("Dataset", [{}])[0]
            bildirectory = dataset.get("bildirectory", "")
            if not bildirectory:
                continue

            match = re.match(r"/bil/data/(.+)", bildirectory)
            if not match:
                continue

            download_path = match.group(1)
            dir_url = f"{DOWNLOAD_BASE}/{download_path}/"

            try:
                req = urllib.request.Request(dir_url)
                with urllib.request.urlopen(req, context=get_ssl_context()) as response:
                    html = response.read().decode()

                for swc_match in re.finditer(r'href="([^"]+\.swc)"', html):
                    filename = swc_match.group(1)
                    if filename.endswith("_reg.swc"):
                        swc_urls.append(f"{DOWNLOAD_BASE}/{download_path}/{filename}")
            except Exception:
                pass

        if len(swc_urls) >= NUM_FILES:
            break

    print(f"Downloading {NUM_FILES} SWC files...")
    for i, url in enumerate(swc_urls[:NUM_FILES]):
        filename = url.split("/")[-1]
        output_path = OUTPUT_DIR / filename
        print(f"  [{i+1}/{NUM_FILES}] {filename}")
        download_file(url, output_path)

    print(f"\nDownloaded files to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()