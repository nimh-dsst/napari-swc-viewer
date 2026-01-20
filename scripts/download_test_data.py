#!/usr/bin/env python
"""Download test SWC files from the Brain Image Library.

This script retrieves a random subset of SWC files from the BIL dataset
DOI 10.35077/g.73 "Morphological diversity of single neurons in molecularly
defined cell types" for testing the napari-swc-viewer plugin.
"""

import json
import random
import re
import urllib.request
import ssl
from pathlib import Path


# BIL API endpoints
API_BASE = "https://api.brainimagelibrary.org"
DOWNLOAD_BASE = "https://download.brainimagelibrary.org"

# Number of SWC files to download
NUM_FILES = 20

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "test_data"

# Known submission UUID for the morphology dataset (from DOI 10.35077/g.73)
MORPHOLOGY_SUBMISSION_UUID = "0fcde5fdd6f7ccb2"


def get_ssl_context():
    """Create SSL context that doesn't verify certificates (BIL API requirement)."""
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


def get_dataset_bildids() -> list[str]:
    """Query the BIL API to get bildids for the SWC morphology dataset."""
    # Query by submission UUID which is specific to this dataset
    result = api_get(f"query/submission?submission_uuid={MORPHOLOGY_SUBMISSION_UUID}")

    if result.get("success") != "true":
        raise RuntimeError(f"API query failed: {result.get('message')}")

    return result.get("bildids", [])


def get_swc_files_for_bildid(bildid: str) -> list[str]:
    """Get list of SWC file URLs for a given bildid."""
    result = api_get(f"retrieve?bildid={bildid}")

    if result.get("success") != "true":
        return []

    swc_urls = []
    for entry in result.get("retjson", []):
        dataset = entry.get("Dataset", [{}])[0]
        bildirectory = dataset.get("bildirectory", "")

        if not bildirectory:
            continue

        # Convert bildirectory to download URL path
        # /bil/data/XX/YY/hash/folder -> XX/YY/hash/folder
        match = re.match(r"/bil/data/(.+)", bildirectory)
        if not match:
            continue

        download_path = match.group(1)

        # List the directory to find SWC files
        dir_url = f"{DOWNLOAD_BASE}/{download_path}/"
        try:
            req = urllib.request.Request(dir_url)
            with urllib.request.urlopen(req, context=get_ssl_context()) as response:
                html = response.read().decode()

            # Parse SWC files from directory listing (exclude _reg.swc for simplicity)
            for swc_match in re.finditer(r'href="([^"]+\.swc)"', html):
                filename = swc_match.group(1)
                if not filename.endswith("_reg.swc"):
                    swc_urls.append(f"{DOWNLOAD_BASE}/{download_path}/{filename}")
        except Exception as e:
            print(f"  Warning: Could not list directory {dir_url}: {e}")

    return swc_urls


def download_file(url: str, output_path: Path) -> bool:
    """Download a file from URL to output_path."""
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, context=get_ssl_context()) as response:
            output_path.write_bytes(response.read())
        return True
    except Exception as e:
        print(f"  Error downloading {url}: {e}")
        return False


def main():
    """Main function to download test SWC files."""
    print("Downloading test SWC files from Brain Image Library")
    print("Dataset: Morphological diversity of single neurons in molecularly defined cell types")
    print("DOI: 10.35077/g.73")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get bildids for the dataset
    print("Querying BIL API for dataset entries...")
    bildids = get_dataset_bildids()
    print(f"Found {len(bildids)} dataset entries")

    # Collect all SWC file URLs
    print("Collecting SWC file URLs...")
    all_swc_urls = []

    # Sample a subset of bildids to search (to avoid querying all 1700+)
    sample_bildids = random.sample(bildids, min(50, len(bildids)))

    for i, bildid in enumerate(sample_bildids):
        print(f"  Checking {bildid} ({i+1}/{len(sample_bildids)})...", end="", flush=True)
        urls = get_swc_files_for_bildid(bildid)
        all_swc_urls.extend(urls)
        print(f" found {len(urls)} SWC files")

        # Stop early if we have enough files to choose from
        if len(all_swc_urls) >= NUM_FILES * 3:
            break

    if len(all_swc_urls) == 0:
        print("Error: No SWC files found!")
        return 1

    print(f"\nTotal SWC files found: {len(all_swc_urls)}")

    # Randomly select files to download
    files_to_download = random.sample(all_swc_urls, min(NUM_FILES, len(all_swc_urls)))

    print(f"\nDownloading {len(files_to_download)} randomly selected SWC files...")

    downloaded = 0
    for i, url in enumerate(files_to_download):
        filename = url.split("/")[-1]
        output_path = OUTPUT_DIR / filename

        print(f"  [{i+1}/{len(files_to_download)}] {filename}...", end="", flush=True)

        if download_file(url, output_path):
            downloaded += 1
            print(" done")
        else:
            print(" failed")

    print(f"\nDownloaded {downloaded} SWC files to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    import urllib.parse
    exit(main())
