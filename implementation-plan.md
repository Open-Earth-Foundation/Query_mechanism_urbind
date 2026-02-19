````markdown
# Implementation Plan: Chroma Index Builder + Persistent Store

This plan implements the **Index Builder / Updater** and the **Chroma Persistent Store** as shown in `docs/architecture/target-architecture.md`.

Goal: create a lightweight, inspectable local vector store for markdown chunks so that later we can plug in a retriever that returns **top-k relevant chunks** and feed those into the existing markdown researcher **without modifying its logic** (we will only swap the input chunks list).

---

## Scope (this iteration)

✅ Build and persist a Chroma index from `documents/**/*.md`  
✅ Incrementally update the index (only re-embed changed content)  
✅ Store rich metadata per chunk (city_name, heading_path, token_count, block_type, etc.)  
✅ Provide simple inspection commands/scripts for debugging chunking and retrieval readiness  
❌ No retriever integration into orchestrator yet (next iteration)  
❌ No query refinement yet (next iteration)

---

## Guiding constraints

- Keep current pipeline intact. The markdown researcher stays unchanged.
- Implement as a self-contained module so later integration is a small diff in orchestrator.
- Optimize for local inspection and debuggability.
- Do not overengineer. Prefer simple scripts and plain JSON manifests.

---

## Deliverables

### 1) New modules

Create a new module namespace:
E.g. below (or what you need)

```text
backend/modules/vector_store/
  __init__.py
  config.py
  models.py
  markdown_blocks.py
  chunk_packer.py
  table_utils.py
  chroma_store.py
  indexer.py
  manifest.py
```
````

### 2) New scripts

Add scripts for one-off build and incremental update:

```text
backend/scripts/
  build_markdown_index.py
  update_markdown_index.py
  inspect_markdown_index.py
```

### 3) New config section

Add config entries to your existing config system (YAML / env / pydantic, whichever you already use):

- `VECTOR_STORE_ENABLED` (default false for safety)
- `CHROMA_PERSIST_PATH` (default `.chroma`)
- `CHROMA_COLLECTION_NAME` (default `markdown_chunks`)
- `EMBEDDING_MODEL` (pick now, can be swapped later) -> openai text-embedding-3-large
- `EMBEDDING_CHUNK_TOKENS` (default 800)
- `EMBEDDING_CHUNK_OVERLAP_TOKENS` (default 80)
- `TABLE_ROW_GROUP_MAX_ROWS` (default 25, used for huge table splitting)
- `INDEX_MANIFEST_PATH` (default `.chroma/index_manifest.json`)

---

## Data model

### Indexed chunk record

Define a stable schema to store in Chroma.

**Chroma fields**

- `id` (string): deterministic chunk id
- `documents` (list[str]): embedding text (retrieval-optimized text)
- `metadatas` (list[dict]): metadata, include raw table or raw text reference
- `embeddings` (list[list[float]]): generated vectors

**Metadata fields (minimum recommended)**

- `city_name`: string (derived from file stem, matches current behavior)
- `source_path`: string (relative path under repo)
- `block_type`: `paragraph | table | list | code`
- `heading_path`: string, formatted `H1 > H2 > H3`
- `chunk_index`: int (per file, stable ordering)
- `token_count`: int (token count for raw_text)
- `content_hash`: string (hash of raw_text)
- `file_hash`: string (hash of full file content)
- `raw_text`: string (evidence text, exact markdown block or chunk)
- `created_at`: ISO string
- `updated_at`: ISO string

Optional but useful:

- `start_line`, `end_line`
- `table_id`, `row_group_index` (if tables are split into groups)

---

## Chunking strategy (index-time only)

We implement **table-aware markdown block parsing**, then pack blocks into token-limited chunks.

### Step A: Parse markdown into blocks

Implement `parse_markdown_blocks(text: str) -> list[MdBlock]`:

Detect blocks:

- Heading: regex `^#{1,6}\s`
- Fenced code: lines starting with ``` or ~~~ until closing fence
- Tables: pipe table detection
  - header line contains `|`
  - next line is a separator row containing `---` and `|`
  - subsequent lines contain `|` until table ends

- Lists: contiguous lines matching bullets `^- `, `^* `, `^\d+\. `
- Paragraph: everything else grouped by blank lines

Maintain a heading stack while scanning lines so each block has:

- `heading_path` list[str]
- `block_type`
- `text`
- line ranges (optional)

### Step B: Pack blocks into chunks

Implement `pack_blocks(blocks, max_tokens, overlap_tokens) -> list[PackedChunk]`:

- Greedily append blocks until token budget reached
- Never split inside code fences
- Never split inside tables unless a table exceeds token budget
- If a table exceeds budget: split by row groups while repeating header+separator rows

### Step C: Dual representation for tables

For each packed chunk:

- `raw_text`: exact markdown evidence
- `embedding_text`: retrieval-optimized text

Rules:

- For non-table chunks: `embedding_text = raw_text` (optionally strip excess whitespace)
- For table chunks: `embedding_text = summarize_table_for_embedding(raw_text)`
  - Deterministic summary:
    - detect title-like first row or preceding heading_path
    - list column names
    - include row count if feasible
    - include first N rows (or first N unique key values)

Store both by placing:

- Chroma `documents` = embedding_text
- metadata `raw_text` = raw_text

---

### Table chunk splitting guarantees

When a markdown table exceeds the embedding chunk token budget, it must be split into row groups while preserving semantic integrity.

The following rules are mandatory:

#### 1. Header repetition

Every table chunk must repeat:

- the header row
- the separator row

This ensures each chunk is a valid standalone markdown table and prevents orphaned rows without column context.

#### 2. Deterministic table identifier

A stable `table_id` must be assigned to all chunks originating from the same source table.

Recommended construction:

---

## Embeddings

Keep embeddings pluggable via a small interface.

### Interface

In `backend/modules/vector_store/models.py`:

- `EmbeddingProvider.embed_texts(texts: list[str]) -> list[list[float]]`

### Provider choices

Implementation should support either:

- local sentence-transformers
- or external API embeddings

Pick one now based on your existing stack. Keep it behind the interface so switching later is one file change.

---

## Manifest and incremental updates

### Manifest format

`.chroma/index_manifest.json` should track:

- `index_version`
- `created_at`, `updated_at`
- `embedding_model`
- `embedding_chunk_tokens`, `embedding_chunk_overlap_tokens`
- `files`: map `{source_path: {file_hash: "...", chunk_ids: [...]}}`

### Update algorithm

For each markdown file:

1. compute `file_hash`
2. compare to manifest
3. if unchanged: skip
4. if changed or new:
   - parse blocks
   - pack into chunks
   - compute `content_hash` per chunk
   - upsert chunk ids into Chroma
   - update manifest entry

5. detect deleted files:
   - delete all chunk_ids from Chroma
   - remove from manifest

Chroma supports deletion by ids, use that for removed chunks.

---

## Chroma persistent store

### Store wrapper

Implement a thin wrapper in `chroma_store.py`:

- `get_client(persist_path) -> chromadb.PersistentClient`
- `get_collection(name) -> Collection`
- `upsert(chunks: list[IndexedChunk])`
- `delete(ids: list[str])`
- `count()`
- `peek(n=5)` (for quick sanity checks)
- `get(where=..., limit=...)` (for inspection)

Use `PersistentClient(path=CHROMA_PERSIST_PATH)` so the index lives on disk.

---

## Scripts

### 1) Build from scratch

`backend/scripts/build_markdown_index.py`

Behavior:

- deletes existing collection or creates new
- builds full index from all `documents/**/*.md`
- writes manifest

CLI flags:

- `--docs-dir documents`
- `--persist-path .chroma`
- `--collection markdown_chunks`
- `--city <optional>` (only index matching city stems)
- `--dry-run` (parse and chunk, print stats, do not embed/store)

Output:

- total files indexed
- total chunks created
- chunk size distribution (token_count min/avg/max)
- number of table chunks vs other

### 2) Incremental update

`backend/scripts/update_markdown_index.py`

Behavior:

- loads manifest
- updates changed/new files
- deletes removed files
- updates manifest

### 3) Inspect index

`backend/scripts/inspect_markdown_index.py`

CLI examples:

- `--city Aarhus --limit 20`
- `--where block_type=table --limit 20`
- `--contains "Climate Neutrality Target" --limit 10` (client-side filter over returned docs)
- `--show-id <chunk_id>` dumps metadata and raw_text

This script is essential for tuning chunking.

---

## Testing and acceptance checks

### Unit tests

Add tests for:

- heading_path stack handling
- table detection on CCC-style tables
- table row-group splitting preserves header rows
- deterministic chunk id generation
- manifest update logic (changed file triggers upsert, unchanged does not)

### Acceptance checklist

- Running build script creates `.chroma/` and `index_manifest.json`
- Index contains records with metadata fields populated
- Tables are not split mid-row in `raw_text`
- `inspect_markdown_index.py` can list chunks for a city and show raw_text
- Re-running update without changes performs near-zero work

---

## Future integration hook (next iteration)

This iteration must expose a retrieval-ready interface:

### Retrieval-ready function signatures

In `indexer.py` or `chroma_store.py`, define stubs that will be used next:

- `ensure_index_up_to_date(docs_dir: str) -> None`
- `retrieve_top_k(query: str, city_filter: list[str] | None, k: int) -> list[RetrievedChunk]`

Where `RetrievedChunk` includes:

- `city_name`
- `raw_text`
- `source_path`
- `heading_path`
- `block_type`
- `score`
- `chunk_id`

### Non-goal now

Do not wire this into orchestrator yet. Just ensure these functions exist and are easy to call later.

---

## Implementation sequence (recommended order)

1. Create `models.py` with `MdBlock`, `PackedChunk`, `IndexedChunk`
2. Implement `markdown_blocks.py` parser (headings, tables, code fences)
3. Implement `chunk_packer.py` with token budget + table row-group splitting
4. Implement deterministic `chunk_id` and `content_hash`
5. Implement `chroma_store.py` wrapper (PersistentClient, get_collection, upsert/delete)
6. Implement `manifest.py` read/write and update logic
7. Implement `indexer.py` (build + update flows)
8. Add scripts (build/update/inspect)
9. Add tests for table-heavy CCC examples and heading_path correctness

---

## Notes

- Keep the markdown researcher unchanged. The only integration change later will be to replace the current chunk list passed into `extract_markdown_excerpts()` with retrieved chunks from Chroma.
- Keep chunking rules stable and deterministic so incremental updates behave predictably.
- Prefer deterministic table summaries for embeddings to avoid hallucinated values.

---
