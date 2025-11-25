#!/usr/bin/env python3
"""
Download ClinicalTrials.gov bulk XML data and upload to S3

This script downloads the complete AllPublicXML.zip file from ClinicalTrials.gov
and uploads it to S3 for future use. The file is approximately 4GB compressed
and 11GB uncompressed.
"""

import os
import sys
import requests
import subprocess
from pathlib import Path
from datetime import datetime

# Constants
BULK_XML_URL = "https://clinicaltrials.gov/AllPublicXML.zip"
S3_BUCKET = "aipi520-antibody-trial-predictor"
S3_KEY = "bulk_data/AllPublicXML.zip"
LOCAL_ZIP_PATH = "data/AllPublicXML.zip"
CHUNK_SIZE = 8192 * 1024  # 8MB chunks for download


def download_with_progress(url, output_path):
    """Download file with progress indicator"""
    print(f"Downloading from {url}")
    print(f"Saving to {output_path}")

    # Create directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Start download
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0

    print(f"Total size: {total_size / (1024**3):.2f} GB")

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

                # Print progress every 100MB
                if downloaded % (100 * 1024 * 1024) < CHUNK_SIZE:
                    progress = (downloaded / total_size) * 100 if total_size > 0 else 0
                    print(f"Progress: {downloaded / (1024**3):.2f} GB / {total_size / (1024**3):.2f} GB ({progress:.1f}%)")

    print(f"✓ Download complete: {output_path}")
    return output_path


def upload_to_s3(local_path, s3_bucket, s3_key):
    """Upload file to S3 using AWS CLI"""
    print(f"\nUploading to S3: s3://{s3_bucket}/{s3_key}")

    s3_uri = f"s3://{s3_bucket}/{s3_key}"

    # Use AWS CLI for reliable upload with progress
    cmd = [
        "aws", "s3", "cp",
        local_path,
        s3_uri,
        "--no-progress"  # Disable progress bar for cleaner CI logs
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error uploading to S3: {result.stderr}")
        sys.exit(1)

    print(f"✓ Upload complete: {s3_uri}")

    # Add metadata with download date
    metadata_cmd = [
        "aws", "s3api", "put-object-tagging",
        "--bucket", s3_bucket,
        "--key", s3_key,
        "--tagging", f"downloaded={datetime.now().strftime('%Y-%m-%d')}"
    ]

    subprocess.run(metadata_cmd, capture_output=True)

    return s3_uri


def check_s3_exists(s3_bucket, s3_key):
    """Check if file already exists in S3"""
    cmd = ["aws", "s3", "ls", f"s3://{s3_bucket}/{s3_key}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def main():
    print("=" * 80)
    print("ClinicalTrials.gov Bulk XML Download")
    print("=" * 80)
    print()

    # Check if already in S3
    if check_s3_exists(S3_BUCKET, S3_KEY):
        print(f"✓ File already exists in S3: s3://{S3_BUCKET}/{S3_KEY}")
        print()
        print("Options:")
        print("  1. Use existing S3 file (recommended)")
        print("  2. Re-download (will take 10-30 minutes)")
        print()

        # For CI/CD, just use existing
        if os.environ.get('CI') or '--force' not in sys.argv:
            print("Using existing S3 file")
            return

    # Check if already downloaded locally
    if Path(LOCAL_ZIP_PATH).exists():
        file_size = Path(LOCAL_ZIP_PATH).stat().st_size / (1024**3)
        print(f"✓ Found local file: {LOCAL_ZIP_PATH} ({file_size:.2f} GB)")
        print("Skipping download, uploading to S3...")
    else:
        # Download the bulk XML file
        print("Starting download of AllPublicXML.zip")
        print("This will take 10-30 minutes depending on network speed...")
        print()

        try:
            download_with_progress(BULK_XML_URL, LOCAL_ZIP_PATH)
        except Exception as e:
            print(f"Error downloading file: {e}")
            sys.exit(1)

    # Upload to S3
    try:
        upload_to_s3(LOCAL_ZIP_PATH, S3_BUCKET, S3_KEY)
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        sys.exit(1)

    print()
    print("=" * 80)
    print("✓ Bulk XML data successfully uploaded to S3")
    print("=" * 80)
    print()
    print(f"S3 Location: s3://{S3_BUCKET}/{S3_KEY}")
    print(f"Local copy: {LOCAL_ZIP_PATH}")
    print()
    print("Next steps:")
    print("  1. Update workflow to download from S3")
    print("  2. Extract and parse XML files")
    print("  3. Filter for Phase 2/3 antibody trials")


if __name__ == "__main__":
    main()
