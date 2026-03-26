---
name: prompt-schema-authoring
description: Mandatory when creating or updating prompt files in this repository. Use it to enforce the required `<role>`, `<task>`, `<input>`, and `<output>` structure with explicit, model-aligned field contracts, plus a prompt-contract drift check against runtime schemas and tests.
---

# prompt-schema-authoring

Use this skill to keep prompts explicit, contract-driven, and context-efficient.

## Triggering rule

- Trigger this skill whenever you create or edit a prompt under `*/prompts/`.
- If the skill is already active for the current turn, do not re-trigger it; just keep following it.

## Workflow

1. Identify the runtime contract before editing the prompt.

- Open the target prompt in `*/prompts/`.
- Open the corresponding model in `app/modules/*/models.py` or any other schema we are using for the LLM input/output definitions.

2. Run a contract parity check before treating the prompt edit as complete.

- Compare prompt `<input>` fields against the payload actually passed in code.
- Compare prompt `<output>` fields against the runtime model, parser, or tool contract.
- Check whether fallback, partial-success, or error-shaped output described in the prompt is still accepted by runtime code.
- If prompt, runtime schema, and tests do not agree, warn the user explicitly and treat the mismatch as required follow-up work, not as a silent prompt-only cleanup.

3. Write prompt sections in this order.

- `<role>`
- `<task>`
- `<input>`
- `<output>`
- Add `<example_output>` whenever possible.

4. Define `<input>` from real runtime payload only.

- List only fields actually passed in code.
- Add type and short purpose for each field.
- Exclude context-junk/internal fields unless the model truly needs them (for example `path`, `chunk_index`, `chunk_count`).

5. Define `<output>` from model contract only.

- State tool invocation requirements explicitly:
  - pass a JSON object or JSON list depending on the tool definition
  - return only the desired output
- Enumerate required and optional fields exactly as expected by the model.
- Explain field behavior clearly.
- Exclude internal/auto fields that should not come from the LLM (for example `created_at`).

6. Add one valid `<example_output>` that conforms to the model.

7. Keep contracts aligned end-to-end.

- If you change prompt output fields, verify models, coercion/parsing, runtime logic, and tests together.
- If the current task is prompt-only or you are not updating the runtime pieces in the same turn, warn the user that contract drift remains and list the exact follow-up changes still required.

8. Require schema-level regression coverage when the contract changes.

- If prompt fields or output contract changed, add or propose at least one schema-level regression test.
- The test should prove that valid structured output still parses and that missing, renamed, or fallback-shaped fields are handled intentionally.

## User-facing warning requirement

- In the final response, state whether the contract parity check passed.
- If it did not pass, include a clear warning that the prompt change is not fully safe until the runtime schema and tests are updated or verified.

## Required Prompt Rules

- Keep instructions explicit and operational.
- Keep output contract field-by-field and typed.
- Avoid asking for wrappers/status/error fields unless the model requires them.
- Avoid asking for timestamps from the LLM.
- Avoid meta phrasing requirements that conflict with downstream synthesis.

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
