#!/usr/bin/env python
"""Check vector store index status."""
import json
from datetime import datetime
from pathlib import Path

manifest_path = Path('.chroma/index_manifest.json')
manifest_mtime = datetime.fromtimestamp(manifest_path.stat().st_mtime)

with open(manifest_path) as f:
    manifest = json.load(f)

files = manifest.get('files', {})
total_chunks = sum(len(f.get('chunk_ids', [])) for f in files.values())

print(f"Manifest last modified: {manifest_mtime}")
print(f"Files indexed: {len(files)}")
print(f"Total chunks: {total_chunks}")
print(f"Build timestamp: {manifest.get('build_timestamp', 'N/A')}")
print(f"Index version: {manifest.get('index_version', 'N/A')}")
print(f"\nFiles in index:")
for fname in sorted(files.keys()):
    chunks = len(files[fname].get('chunk_ids', []))
    print(f"  {fname}: {chunks} chunks")

chroma_db = Path('.chroma/chroma.sqlite3')
if chroma_db.exists():
    db_size = chroma_db.stat().st_size / (1024 * 1024)
    db_mtime = datetime.fromtimestamp(chroma_db.stat().st_mtime)
    print(f"\nChroma DB: {db_mtime}")
    print(f"Size: {db_size:.2f} MB")
