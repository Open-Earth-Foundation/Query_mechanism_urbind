"""Calculate total tokens from run log."""

import re
from pathlib import Path

log_path = Path('output/20260205_2352/run.log')
content = log_path.read_text(encoding='utf-8')

# Find all input_tokens and output_tokens
input_matches = re.findall(r'"input_tokens": (\d+)', content)
output_matches = re.findall(r'"output_tokens": (\d+)', content)

total_input = sum(int(x) for x in input_matches)
total_output = sum(int(x) for x in output_matches)
total_tokens = total_input + total_output

print(f'Total input tokens: {total_input:,}')
print(f'Total output tokens: {total_output:,}')
print(f'Total tokens: {total_tokens:,}')
print(f'')
print(f'Cost estimate (Grok-4.1-fast on OpenRouter):')
print(f'  Input: ${total_input * 0.16 / 1_000_000:.2f}')
print(f'  Output: ${total_output * 0.64 / 1_000_000:.2f}')
print(f'  Total: ${(total_input * 0.16 + total_output * 0.64) / 1_000_000:.2f}')
