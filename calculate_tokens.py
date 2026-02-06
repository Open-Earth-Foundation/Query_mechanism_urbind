"""
Calculate token usage for documents folder.

Brief: Analyze tiktoken count for all markdown files in documents/ root
Inputs: None (reads from documents/ folder)
Outputs: Token count summary and cost analysis

Usage (from project root):
- python calculate_tokens.py
"""

from __future__ import annotations

import tiktoken
from pathlib import Path

# Initialize tiktoken encoder for gpt-4 (cl100k_base is used by grok-4)
encoding = tiktoken.get_encoding("cl100k_base")

docs_dir = Path('documents')
md_files = sorted([f for f in docs_dir.glob('*.md') if f.is_file()])

print(f"Calculating tokens for {len(md_files)} markdown files in documents/\n")
print(f"{'File':<25} {'Tokens':>10} {'KB':>10}")
print("-" * 47)

total_tokens = 0
file_tokens = []
for f in md_files:
    content = f.read_text(encoding='utf-8')
    tokens = len(encoding.encode(content))
    size_kb = f.stat().st_size / 1024
    total_tokens += tokens
    file_tokens.append((f.name, tokens, size_kb))
    print(f"{f.name:<25} {tokens:>10,} {size_kb:>10.1f}")

print("-" * 47)
print(f"{'TOTAL':<25} {total_tokens:>10,}")

# Now estimate what we're sending to the API
print(f"\n{'='*70}")
print(f"COST ANALYSIS (Grok-4.1-fast on OpenRouter)")
print(f"{'='*70}")

# Read the system prompt
prompt_path = Path('app/prompts/markdown_researcher_system.md')
system_prompt = prompt_path.read_text(encoding='utf-8')
system_tokens = len(encoding.encode(system_prompt))

num_chunks = 73  # from the log: "Processing 73 markdown batches"
avg_tokens_per_chunk = total_tokens // num_chunks

print(f"\nSystem prompt tokens: {system_tokens:,}")
print(f"Total document tokens: {total_tokens:,}")
print(f"Number of chunks (batches): {num_chunks}")
print(f"Average tokens per chunk: {avg_tokens_per_chunk:,}")

tokens_per_chunk_request = system_tokens + avg_tokens_per_chunk
total_per_run = num_chunks * tokens_per_chunk_request

print(f"\nTokens per chunk request:")
print(f"  System prompt: {system_tokens:,}")
print(f"  Average chunk content: {avg_tokens_per_chunk:,}")
print(f"  Total: {tokens_per_chunk_request:,}")

print(f"\nPer complete run (all {num_chunks} chunks):")
print(f"  Input tokens: ~{total_per_run:,}")
print(f"  Approx output tokens (per chunk ~1500): ~{num_chunks * 1500:,}")
print(f"  Total tokens per run: ~{total_per_run + (num_chunks * 1500):,}")

# Cost estimation (Grok-4.1-fast pricing)
cost_per_1m_input = 0.16  # OpenRouter pricing (may vary)
cost_per_1m_output = 0.64
total_cost = (total_per_run * cost_per_1m_input / 1_000_000) + (num_chunks * 1500 * cost_per_1m_output / 1_000_000)

print(f"\nEstimated cost per run:")
print(f"  Input: ${total_per_run * cost_per_1m_input / 1_000_000:.4f}")
print(f"  Output: ${num_chunks * 1500 * cost_per_1m_output / 1_000_000:.4f}")
print(f"  Total: ${total_cost:.4f}")

# Breakdown by city
print(f"\n{'='*70}")
print(f"TOP 10 LARGEST CITIES (by token count)")
print(f"{'='*70}")
sorted_files = sorted(file_tokens, key=lambda x: x[1], reverse=True)
for i, (name, tokens, kb) in enumerate(sorted_files[:10], 1):
    pct = (tokens / total_tokens) * 100
    print(f"{i}. {name:<25} {tokens:>10,} tokens ({pct:>5.1f}%)")
