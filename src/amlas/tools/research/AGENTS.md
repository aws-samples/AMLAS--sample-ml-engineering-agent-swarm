# src/amlas/tools/research

## Purpose

Aggregate research-provider results and bounded URL retrieval with explicit source/coverage information.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Keep provider/source provenance and bounded coverage in returned research records.
- HTTP(S) URL retrieval retains public-address checks, bounded fetches and HTTPS certificate verification.
- Registered research_fetch_urls and research_search projections fit a fixed
  8000-byte compact UTF8 ToolResult.content budget (lowered from 60000 on
  2026-11-07: the three qwen9b_core_v1 research units returned about
  64 KB of the 160 KiB normal-run SFT checkpoint DATA cap, and one run reached its
  first SFT call with a conversation over that cap; the cap itself stays), including the SDK's
  default-JSON TEXT wrapper and nested supplied pages/provider metadata. The count
  uses the ASCII-escaped `json.dumps` text that Strands 1.42.0 builds. The locked
  Strands 1.65.0 leaves non-ASCII characters unescaped, which never takes more
  bytes: there the count is an upper bound, exact for ASCII-only projections. Shorten
  only declared page-text/search-snippet prefixes; retain whole URLs/canonical
  references, metadata and provider failures. Returned/available lengths and
  partial coverage stay exact. If whole metadata alone cannot fit, refuse with
  a finite explicit budget error; never rewrite refs, hide failure or raise caps.
- Binary acquisition uses a separate streamed transport, capped at 512 MiB and
  one 900-second deadline with five individually validated HTTPS redirect hops.
  Remote-fetch flags and host allowlists apply to every hop; every pinned address
  passes the full nonpublic/multicast/global predicate. No proxies or credentials.
- Redirect/error/partial-response bodies are rejected before reading; expected
  size and payload SHA-256/Git-framed SHA-1 are retained without signed CDN URLs.
  Small JSON metadata has its own cap; existing text budgets remain unchanged.
- DNS validation uses one-way anonymous `Pipe(duplex=False)` with bounded ASCII
  JSON through `send_bytes`/nonblocking `recv_bytes` only. No Queue, pickle,
  socketpair, named IPC or shared-memory allocation is part of this resolver.
  Payload4092 plus the measured CPython four-byte header must fit PC_PIPE_BUF;
  reject an incompatible pipe before helper spawn. Strict parent decoding bounds
  bytes/depth/IP count/literal syntax and rejects duplicates/nonfinite tokens,
  schema drift and every nonpublic address before pinning. No raw exception or
  URL/credential detail crosses IPC. Require normal helper exit and one absolute
  deadline before/after poll/read/join/cleanup; partial frames fail promptly.
  Close both ends and retain terminate/join/kill escalation with reaped-handle
  release. The fixed fork child exits explicitly only after bounded packet send
  and writer cleanup, avoiding inherited executor thread-shutdown hooks. Send
  or close failure retains nonzero status; the parent still requires exit0 and
  valid public data. No private threading-state mutation or nonzero-exit allowance. Existing multiprocessing stdin /dev/null read remains an independently
  measured runtime dependency; this contract adds no filesystem or mount grant.
- `aggregator.py:MergedResult` — A ResearchResult enriched with the set of sources that contributed it.
- `aggregator.py:research_aggregate` — Fan out `query` to all sources concurrently, dedupe, rank.
- `arxiv.py:ArxivError` — Raised on HTTP failure or unparseable Atom response.
- `arxiv.py:parse_arxiv_atom` — Parse an arxiv Atom-feed response into normalized ResearchResults.
- Owned files: `__init__.py`, `aggregator.py`, `arxiv.py`, `papers_with_code.py`, `public_fetch.py`, `semantic_scholar.py`, `strands_tools.py`, `tavily.py`, `types.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
