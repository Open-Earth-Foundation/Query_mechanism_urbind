---
name: prompt-schema-authoring
description: Create or update any LLM-related prompt file in this repository using the required `<role>`, `<task>`, `<input>`, and `<output>` structure, with optional `<tools>` when tool policy is needed, and explicit model-aligned field contracts. Use for prompts stored in markdown, Python prompt templates, YAML/JSON config prompts, and any other runtime prompt definitions for large language models (LLMs).
---

# prompt-schema-authoring

Use this skill to keep prompts explicit, contract-driven, and context-efficient.

## Workflow

1. Identify the runtime contract before editing the prompt.

- Open the target prompt in `*/prompts/`.
- Open the corresponding model in `app/modules/*/models.py` or any other schema we are using for the LLM input/output definitions.
- Open the real payload builder or prompt composition code that populates the prompt input.
- Open the runtime tool registration code when the prompt can call tools.

2. Run a contract parity check before treating the prompt edit as complete.

2. Write prompt sections in this order.

- `<role>`
- `<task>`
- `<input>`
- `<tools>` (required whenever the prompt can call one or more tools)
- `<output>`
- Add `<example_output>` whenever possible.

3. Define `<input>` from real runtime payload only.

- List only fields actually passed in code.
- Add type and short purpose for each field.
- Exclude context-junk/internal fields unless the model truly needs them (for example `path`, `chunk_index`, `chunk_count`).

5. Define `<tools>` when tool selection or policy exists.

- Add `<tools>` for every tool-capable prompt. Do not treat it as optional.
- List each tool and when to use it.
- List when not to use it.
- Keep user-facing formatting rules in `<output>`, not `<tools>`.
- Keep exact argument schemas in one place: `<output>`. In `<tools>`, focus on usage policy.

6. Define `<output>` from model contract only.

- State tool invocation requirements explicitly:
  - pass a JSON object or JSON list depending on the tool definition
  - return only the desired output
- Enumerate required and optional fields exactly as expected by the model.
- Explain field behavior clearly.
- Exclude internal or auto fields that should not come from the LLM (for example `created_at`).

7. Add one valid `<example_output>` that conforms to the model.

8. Run mandatory post-edit verification.

- Re-open the prompt after editing and confirm the required sections are present.
- If the prompt can call tools, confirm `<tools>` exists and that the tool names match the runtime tool registration exactly.
- Confirm every runtime input field exposed to the model is named in `<input>`.
- Confirm every LLM-produced field from the output model is named in `<output>` unless the runtime fills it programmatically, in which case say that explicitly in the prompt.
- Add or update a prompt-contract regression test in the same change.

9. Keep contracts aligned end-to-end.

- If you change prompt output fields, update models, coercion or parsing, runtime logic, and tests in the same change.
- If you change prompt input fields, update the payload builder, prompt text, and prompt-contract tests in the same change.
- If the prompt uses tools, update the prompt, tool registration, and tool-focused assertions together.

10. Require schema-level regression coverage when the contract changes.

- If prompt fields or output contract changed, add or propose at least one schema-level regression test.
- The test should prove that valid structured output still parses and that missing, renamed, or fallback-shaped fields are handled intentionally.

## User-facing warning requirement

- In the final response, state whether the contract parity check passed.
- If it did not pass, include a clear warning that the prompt change is not fully safe until the runtime schema and tests are updated or verified.

## Required Prompt Rules

- Keep instructions explicit and operational.
- Keep output contract field-by-field and typed.
- Keep required prompt blocks: `<role>`, `<task>`, `<input>`, `<output>`.
- Add `<tools>` for tool-capable prompts, and use it for tool usage policy.
- Treat missing `<tools>` on a tool-capable prompt as a contract bug.
- Avoid asking for wrappers, status, or error fields unless the model requires them.
- Avoid asking for timestamps from the LLM.
- Avoid meta phrasing requirements that conflict with downstream synthesis.

## Prompt-Contract Test Template

When you edit a prompt, add or update a focused regression test that checks the prompt against the real runtime contract.

Minimum assertions:

- required sections exist: `<role>`, `<task>`, `<input>`, `<output>`, and `<example_output>`
- `<tools>` exists for tool-capable prompts
- every runtime input field name appears in `<input>`
- every output model field name appears in `<output>`, except fields the runtime sets programmatically
- the expected tool name appears in the prompt when the runtime uses tools

Example pattern:

```python
def test_example_prompt_matches_runtime_contract() -> None:
    prompt = PROMPT_PATH.read_text(encoding="utf-8")

    for section in ("<role>", "<task>", "<input>", "<tools>", "<output>", "<example_output>"):
        assert section in prompt

    assert "submit_example_output" in prompt

    for field_name in ExamplePromptInput.model_fields:
        assert f"`{field_name}`" in prompt

    for field_name in ExamplePromptOutput.model_fields:
        if field_name in {"runtime_only_field"}:
            continue
        assert f"`{field_name}`" in prompt
```

## Prompt Skeleton

```md
<role>
...
</role>

<task>
...
</task>

<input>
Input is a JSON object with:
- `field_name` (type): purpose
</input>

<tools>
Available tools:
- `tool_name`: when to use, when not to use.
</tools>

<output>
You must call tool `tool_name` and pass a JSON object (not a JSON string).
Return only that tool call.

The tool argument must match `ModelName`:

- `field_name` (...)
</output>

<example_output>
{
"field_name": "..."
}
</example_output>
```