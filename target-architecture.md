```mermaid
flowchart TD
  U[User] --> FE[Frontend Document Builder]
  FE --> API[Backend API]
  API --> ORCH[Orchestrator run pipeline]

  ORCH -->|optional| QR[Query Refiner]
  QR --> RET[Retriever]
  ORCH -->|without query refiner| RET

  RET -->|top k markdown chunks| MR[Markdown Researcher]
  MR --> EX[Structured Excerpts]
  EX --> WR[Writer]
  WR --> RUN[(Run Store)]
  RUN --> OUT[Generated Document]

  OUT --> CHATBTN[Chat About the Answer]
  CHATBTN --> CHATUI[Context Chat Workspace]
  CHATUI --> CHATAPI[Run Scoped Chat API]
  CHATAPI --> MEM[(Chat Memory Store)]
  CHATAPI --> LOAD[Load saved final document and context bundle]
  LOAD --> RUN
  LOAD --> CHATSRV[Context Chat Service]
  CHATSRV --> LLM[LLM reply generation with citations and calculator tools]
  LLM --> CHATAPI
  CHATAPI --> CHATUI

  CHATAPI -->|optional one city follow up search| FROUTE[Chat Follow Up Router]
  FROUTE --> FRET[Focused City Retrieval]
  FRET --> FB[(Chat Owned Follow Up Bundle)]
  FB --> LOAD

  subgraph VS[Vector Store Layer]
    IDX[Index Builder]
    CH[(Chroma Store)]
    IDX --> CH
    RET --> CH
    FRET --> CH
  end

  subgraph DOCS[Markdown Corpus]
    MD[Markdown Documents]
  end

  MD --> IDX
```

Notes:

- The product remains document-first: chat starts only after a completed run has produced a saved final document and context bundle.
- Context chat is run-scoped. It reuses persisted run artifacts, stores conversation state separately in chat memory, and can combine multiple completed run contexts in one session.
- When the chat router needs narrower evidence, it can trigger a focused one-city follow-up retrieval and attach that result back into the active chat session as a chat-owned follow-up bundle.
- When a chat turn is predicted to require overflow map-reduce, the API now persists the user message, queues a split-mode chat job, and the frontend polls job status until the final assistant message is appended to chat memory.
