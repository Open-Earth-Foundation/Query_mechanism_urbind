import re

with open(r'd:\GitHub\Query_mechanism_urbind\output\20260209_0043\run.log', 'r', encoding='utf-8') as f:
    content = f.read()

input_tokens = [int(t) for t in re.findall(r'"input_tokens": (\d+)', content)]
output_tokens = [int(t) for t in re.findall(r'"output_tokens": (\d+)', content)]
reasoning_tokens = [int(t) for t in re.findall(r'"reasoning_tokens": (\d+)', content)]

print(f'Total API calls: {len(input_tokens)}')
print(f'Total input tokens: {sum(input_tokens):,}')
print(f'Total output tokens: {sum(output_tokens):,}')
print(f'Total reasoning tokens: {sum(reasoning_tokens):,}')
print(f'Avg input/call: {sum(input_tokens)//len(input_tokens):,}')
print(f'Avg output/call: {sum(output_tokens)//len(output_tokens):,}')
print(f'')
print(f'Cost estimate (OpenRouter Grok-4.1-fast):')
input_cost = sum(input_tokens) / 1_000_000 * 0.16
output_cost = sum(output_tokens) / 1_000_000 * 0.64
print(f'  Input: ${input_cost:.4f}')
print(f'  Output: ${output_cost:.4f}')
print(f'  Total: ${input_cost + output_cost:.4f}')
