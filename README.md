# Cred Domain Support Agent

**Track: Banking & FinTech (Cred)**

A production-minded **LangGraph-orchestrated support agent** for a fictional lending company, Cred.

The agent answers loan-policy questions from a locally indexed knowledge base, looks up loan-application status from a deterministic dataset, maintains persisted conversation memory, validates structured responses, applies input/output guardrails, exposes the status tool through MCP, supports SQLite checkpointing and resilience demonstrations, provides a FastAPI deployment, and is evaluated with deterministic RAG-triad scoring.

The entire graded workflow is designed to run with:

```text
MOCK_LLM=true
```

with **zero API keys and zero network calls required** for the deterministic execution path.

---

# 1. Capstone Overview

Cred's lending-operations team needs a support agent capable of:

* answering loan-policy questions from a controlled knowledge base;
* checking the status of a specific loan application;
* remembering conversation context within a thread;
* safely handling fixed-format PII;
* detecting prompt-injection attempts;
* refusing unsupported questions when retrieved context is insufficient;
* returning every response through a validated structured schema;
* exposing the status lookup tool through the Model Context Protocol;
* recovering from interruptions using SQLite checkpoints;
* recovering from transient failures with retry logic;
* enforcing per-node and global execution timeouts;
* serving the agent through FastAPI;
* producing structured request logs;
* evaluating retrieval/generation behavior at scale under deterministic `MOCK_LLM`.

---

# 2. Part 1 — Dataset Design & RAG Core

## Deterministic Dataset

The canonical dataset is generated in:

```text
dataset.py
```

The dataset is exposed as:

```text
LOAN_APPLICATIONS
```

### Dataset design choices

| Choice                              | Value                                  |
| ----------------------------------- | -------------------------------------- |
| Initial seed                        | `42`                                   |
| Number of records                   | `50`                                   |
| Fraud-review generation probability | `0.20`                                 |
| Fraud-review acceptance range       | `10%–30%`                              |
| Generation method                   | deterministic seeded random generation |
| Record IDs                          | `LA-0001` through `LA-0050`            |

The generator begins with seed `42`. If a candidate dataset fails validation, the generator deterministically increments the seed and regenerates rather than manually editing records.

The verified dataset was generated successfully on the first attempt.

### Category generation

The required categories are:

```text
Personal Loan
Home Loan
Auto Loan
Education Loan
Business Loan
```

The verified dataset contains:

```text
Personal Loan:   15
Home Loan:        8
Auto Loan:       11
Education Loan:   8
Business Loan:    8
```

### Status generation weights

The status vocabulary and generation weights are:

```text
Submitted:    15%
Under Review: 20%
Approved:     25%
Rejected:     15%
Disbursed:    25%
```

These are generation probabilities, not requirements for the final observed percentages.

The verified final status counts were:

```text
Submitted:     9
Under Review:  9
Approved:      8
Rejected:      8
Disbursed:    16
```

### Loan amount ranges

Loan amounts are generated according to category:

```text
Personal Loan:   ₹50,000 – ₹15,00,000
Home Loan:       ₹20,00,000 – ₹1,50,00,000
Auto Loan:       ₹3,00,000 – ₹30,00,000
Education Loan:  ₹1,00,000 – ₹30,00,000
Business Loan:   ₹5,00,000 – ₹1,00,00,000
```

These ranges reflect realistic relative borrowing scales: personal loans are generally smaller unsecured loans, home loans are substantially larger, and auto, education, and business loans occupy intermediate or higher ranges appropriate to their purposes.

Generated amounts are rounded to the nearest ₹5,000.

### Fraud-review validation

The verified dataset contains:

```text
11 / 50 records
22.00%
```

with `flagged_for_fraud_review=True`.

This satisfies the required `10%–30%` band.

### Dataset verification

Run:

```text
python dataset.py
```

The resulting validation output is preserved in:

```text
transcripts/part1_dataset_demo.txt
```
---

# 3. Deterministic Execution Guarantee

The project is designed around deterministic `MOCK_LLM` execution.

The environment configuration is:

```text
MOCK_LLM=true
```

No hosted language model API key is required.

The RAG embeddings use the local `all-MiniLM-L6-v2` SentenceTransformers model, and ChromaDB stores the vector indexes locally. The MCP server also runs locally through FastMCP.

The graded workflow therefore does not require:

* OpenAI API keys;
* paid accounts;
* credit cards;
* hosted model access;
* external databases;
* external MCP services;
* internet access during normal execution.

The project must be run in an environment where the required local embedding model and Python dependencies are already available.

---

# 4. Knowledge Base

The knowledge base is stored under:

```text
knowledge_base/
```

It contains 12 original policy documents:

```text
01_loan_eligibility.md
02_emi_calculation.md
03_credit_card_fees.md
04_kyc_requirements.md
05_fraud_dispute_resolution.md
06_account_closure.md
07_interest_rate_slabs.md
08_prepayment_penalties.md
09_minimum_balance.md
10_credit_score_factors.md
11_joint_account_rules.md
12_nri_account_eligibility.md
```

The manifest is:

```text
knowledge_base/manifest.json
```

The documents cover every required knowledge-base topic from the capstone brief.

---

# 5. Chunking, Embeddings & ChromaDB

Both required chunking strategies are implemented in:

```text
rag/chunking.py
```

### Strategies

```text
fixed
sentence
```

The project uses the local SentenceTransformers model:

```text
all-MiniLM-L6-v2
```

Two separate persistent ChromaDB collections are used:

```text
cred_kb_fixed
cred_kb_sentence
```

Verified index sizes:

```text
fixed:     18 chunks | 12 documents
sentence:  13 chunks | 12 documents
```

The strategy-to-collection mapping is:

```text
fixed     -> cred_kb_fixed
sentence  -> cred_kb_sentence
```

---

# 6. Similarity Calibration & Grounded Generation

Retrieval is implemented in:

```text
rag/retrieval.py
```

Grounded generation is implemented in:

```text
rag/generation.py
```

The recommended production strategy is:

```text
sentence
```

using:

```text
cred_kb_sentence
```

with:

```text
top_k = 3
```

## 6.1 Empirical threshold calibration

The threshold was not chosen from a generic tutorial default.

The calibration used five in-scope queries:

```text
What documents are required for KYC?
How is EMI calculated?
What factors affect a credit score?
How do I close my account?
Are there penalties for prepaying a loan?
```

and two deliberately out-of-scope queries:

```text
What is the weather today?
Who won the cricket match?
```

Observed top-1 similarities:

| Query           |     Fixed |  Sentence |
| --------------- | --------: | --------: |
| KYC             |  0.493123 |  0.486820 |
| EMI             |  0.461034 |  0.483376 |
| Credit score    |  0.482284 |  0.482284 |
| Account closure |  0.127687 |  0.132282 |
| Prepayment      |  0.092424 |  0.092424 |
| Weather         | -0.778358 | -0.816783 |
| Cricket         | -0.792805 | -0.805880 |

The lowest observed in-scope similarity was:

```text
0.092424
```

The highest observed out-of-scope similarity was:

```text
-0.778358
```

The frozen threshold is:

```text
SIMILARITY_THRESHOLD = 0.05
```

Therefore:

```text
-0.778358 < 0.05 < 0.092424
```

The threshold was selected from the observed calibration results rather than from an arbitrary `0.5`, `0.6`, or `0.7` tutorial value.

## 6.2 Grounded generation behavior

For each query:

1. retrieve the top three chunks;
2. inspect the highest-ranked result;
3. compare its similarity against `0.05`;
4. treat the query as unsupported when the top result is below the threshold;
5. otherwise construct the answer only from the top-ranked retrieved chunk.

The exact unsupported fallback is:

```text
I don't know based on the Cred knowledge base.
```

Phase 1 demonstrations are preserved in:

```text
transcripts/part1_similarity_calibration.txt
transcripts/part1_grounded_generation.txt
```

---

# 7. Precision@3 / Recall@3 Evaluation

Retrieval evaluation is implemented in:

```text
eval/retrieval_eval.py
```

Evaluation is performed at the **parent-document level**, meaning multiple chunks belonging to the same document are deduplicated before scoring.

For Precision@3:

```text
relevant retrieved parent documents / 3
```

For Recall@3:

```text
relevant retrieved parent documents / number of relevant parent documents
```

The same five evaluation queries were used for both strategies.

### Verified results

| Strategy       | Mean Precision@3 | Mean Recall@3 |
| -------------- | ---------------: | ------------: |
| Fixed-size     |           0.3333 |        1.0000 |
| Sentence-based |           0.3333 |        1.0000 |

Both strategies retrieved the relevant parent document at rank 1 for all five queries.

### Recommendation

Neither strategy demonstrated a measured retrieval-performance advantage: both achieved Precision@3 of `0.3333` and Recall@3 of `1.0000`. Sentence-based chunking was therefore selected as the deployment strategy because sentence-level chunks provide natural text units for grounded answer construction, not because they achieved better retrieval metrics.

The full retrieval demonstration is preserved in:

```text
transcripts/part1_retrieval_eval.txt
```

---

# 8. Part 2 — LangGraph Agent

The frozen Part 2 agent is implemented under:

```text
agent/
```

Important files:

```text
agent/tools.py
agent/graph.py
agent/memory.py
agent/schema.py
agent/guardrails.py
```

The graph contains these nodes:

```text
guard_input
load_memory
classify_intent
rag_answer
status_lookup
format_response
persist_memory
```

The high-level flow is:

```text
START
  -> guard_input
  -> load_memory
  -> classify_intent
  -> RAG OR status lookup
  -> format_response
  -> persist_memory
  -> END
```

The conditional routing is deterministic.

A query containing a record ID matching:

```text
LA-\d{4}
```

together with status/application language is routed to the status tool.

Other supported policy questions are routed to RAG.

---

# 9. Status Lookup & Escalation Scoring

The status tool is:

```text
agent/tools.py
```

with the interface:

```text
check_loan_application_status(record_id: str)
```

It reads from the canonical:

```text
LOAN_APPLICATIONS
```

and does not create a second dataset.

## 9.1 Escalation formula

The escalation score combines fraud-review status with a normalized recency signal.

```text
fraud_signal = 1 if flagged_for_fraud_review else 0

recency_signal = 1 - (days_since_created / 30)

escalation_score =
    0.55 * fraud_signal +
    0.45 * recency_signal
```

The score is clamped to:

```text
[0, 1]
```

The constants are:

```text
FRAUD_WEIGHT = 0.55
RECENCY_WEIGHT = 0.45
MAX_DAYS_SINCE_CREATED = 30
```

## 9.2 Dataset-calibrated threshold

The threshold is calibrated from the 25th percentile of the actual generated dataset:

```text
P25(days_since_created) = 8.0
```

Therefore:

```text
P25 recency signal
= 1 - (8 / 30)
= 0.733333...
```

The threshold is:

```text
0.45 * 0.733333...
= 0.33
```

The escalation decision is:

```text
should_escalate = escalation_score >= 0.33
```

This produces a designed continuous escalation score rather than a bare boolean fraud flag.

The verified demonstration covers:

* flagged application above threshold;
* unflagged application above threshold;
* unflagged application below threshold;
* unknown record.

Verification:

```text
transcripts/part2_tool_demo.txt
```

---

# 10. Persistent Conversation Memory

Memory is implemented in:

```text
agent/memory.py
```

and persisted under:

```text
transcripts/memory/
```

Each thread stores its conversation history as JSON containing:

```text
thread_id
messages
```

The graph loads memory before intent classification and persists the completed user/assistant exchange after response formatting.

A follow-up can reuse the most recent loan-application ID from the same thread.

A fresh thread begins with no prior history.

Demonstrations:

```text
transcripts/part2_memory_multiturn.txt
transcripts/part2_memory_fresh.txt
```

---

# 11. Structured Output

The public structured response is defined in:

```text
agent/schema.py
```

The response fields are:

```text
answer
route
escalation_score
thread_id
record_id
grounded
```

Validation requires:

* `answer` to be non-empty;
* `route` to be either `rag` or `status`;
* `escalation_score` to be null or within `[0, 1]`;
* `thread_id` to be non-empty;
* `record_id` to be a string or null;
* `grounded` to be boolean or null;
* unexpected fields to be rejected.

The graph validates the structured response before returning it through:

```text
structured_response
```

Structured-output validation is demonstrated in the Phase 2 verification transcripts.

---

# 12. Guardrails

Guardrails are implemented in:

```text
agent/guardrails.py
```

## 12.1 PII masking

The input-side guardrail masks:

```text
PAN
Grouped Aadhaar
Bank account numbers
```

using:

```text
[PAN_MASKED]
[AADHAAR_MASKED]
[BANK_ACCOUNT_MASKED]
```

The original unmasked query is not persisted as the conversation's user message.

The fixed-format PII scope deliberately does not claim reliable masking of arbitrary applicant names or free-form income values.

## 12.2 Prompt-injection detection

Instruction-override attempts such as:

```text
Ignore previous instructions
Reveal your system prompt
```

are detected and blocked.

The blocked response is:

```text
Prompt injection detected.
```

## 12.3 RAG groundedness

The output-side groundedness guardrail reuses the frozen Phase 1 `grounded` result.

It does not introduce a second similarity calculation or a replacement threshold.

Unsupported RAG requests return:

```text
I don't know based on the Cred knowledge base.
```

All three guardrails are demonstrated in:

```text
transcripts/part2_guardrails_demo.txt
```

---

# 13. Part 3 — FastAPI Deployment

The FastAPI application is implemented under:

```text
api/
```

with:

```text
api/main.py
```

The backend exposes:

```text
GET /health
POST /ask
```

Pydantic request/response models are used.

The `/ask` endpoint invokes the frozen Phase 2 graph rather than reimplementing the agent.

The normal invocation path consumes:

```text
result["structured_response"]
```

from the existing `agent.graph.run_query(...)` interface.

## API verification

The verified health response was:

```json
{
  "status": "ok",
  "service": "cred-domain-support-agent"
}
```

A verified `/ask` request for:

```text
How is EMI calculated?
```

returned a grounded RAG response through the LangGraph agent.

Demonstration:

```text
transcripts/part3_fastapi_demo.txt
```

---

# 14. Structured JSONL Observability

Observability is implemented in:

```text
observability/logging.py
```

and API middleware in:

```text
api/main.py
```

Request logs are written to:

```text
transcripts/logs/api_requests.jsonl
```

Each request produces one JSON-Lines entry containing structured request metadata, including:

* trace ID;
* timing information;
* masked request text;
* request/response status information;
* result metadata.

The same `mask_pii()` behavior used by the agent is applied before request text reaches disk.

A deliberate PII test verified masking of:

```text
PAN
Aadhaar
bank account number
```

before logging.

Demonstration:

```text
transcripts/part3_logging_demo.txt
```

---

# 15. Part 3 — RAG Triad Evaluation

The evaluation dataset is:

```text
eval/part3_eval_dataset.py
```

It contains exactly:

```text
15 queries
```

Coverage:

```text
12 required knowledge-base topics
2 deliberately out-of-scope queries
1 status/edge-case query
```

The dataset validation produced:

```text
Total queries: 15
Knowledge-base topics covered: 12
Out-of-scope queries: 2
Dataset validation: PASS
```

The RAG-triad evaluator is:

```text
observability/rag_triad.py
```

It evaluates:

```text
Context Relevance
Groundedness
Answer Relevance
```

for every query using a deterministic `MOCK_LLM` judge.

### Verified configuration

```text
MOCK_LLM: True
Judge: deterministic MockLLMJudge
Strategy: sentence
Top-k: 3
Similarity threshold: 0.05
Queries evaluated: 15
```

### Overall averages

```text
Context Relevance: 0.65
Groundedness:      0.67
Answer Relevance:  0.68
```

The complete per-query evaluation and averages are preserved in:

```text
transcripts/part3_rag_triad_demo.txt
```

---

# 16. Part 4 — MCP Interoperability

The MCP implementation is under:

```text
mcp_server/
```

with:

```text
mcp_server/server.py
mcp_server/client.py
```

The project uses:

```text
fastmcp
```

The local MCP server exposes:

```text
lookup_loan_application_status
```

which wraps the existing frozen:

```text
check_loan_application_status
```

implementation.

The server uses FastMCP HTTP transport and is mounted at:

```text
http://127.0.0.1:8000/mcp
```

The MCP client is a separate file and separate process from the LangGraph agent.

The verified client called the tool successfully for:

```text
LA-0001
LA-0002
```

Both returned successful standardized MCP tool results.

The demonstration is preserved in:

```text
transcripts/part4_mcp_demo.txt
```

### MCP server

From the project root:

```text
python mcp_server/server.py
```

The server listens locally on:

```text
http://127.0.0.1:8000/mcp
```

### MCP client

With the server running in a separate terminal:

```text
python mcp_server/client.py
```

The client connects to `/mcp`, lists available tools, and calls the status tool for the two configured record IDs.

---

# 17. Part 4 — SQLite Checkpointing

The checkpointing dependency is:

```text
langgraph-checkpoint-sqlite
```

The demonstration is:

```text
eval/part4_checkpoint.py
```

The SQLite checkpoint database is:

```text
checkpoints.sqlite
```

The demonstration uses the thread:

```text
phase4-checkpoint-demo
```

The run contains four nodes:

```text
node_a
node_b
node_c
node_d
```

The first execution deliberately stops after:

```text
node_b
```

The checkpointed state shows that:

```text
node_a_result: Node A completed.
node_b_result: Node B completed.
node_c_result:
node_d_result:
```

The same thread ID is then resumed.

The resumed run executes:

```text
node_c
node_d
```

The final execution log verifies:

```text
node_a executed
node_b executed
node_c executed
node_d executed
```

with execution counts:

```text
node_a: 1
node_b: 1
node_c: 1
node_d: 1
```

The demonstration explicitly verifies that `node_a` and `node_b` were not re-executed after resumption.

Transcript:

```text
transcripts/part4_checkpoint_demo.txt
```

---

# 18. Part 4 — Timeouts & Retries

The resilience demonstration is:

```text
eval/part4_resilience.py
```

## Retry policy

The demonstrated retry policy uses:

```text
max_attempts = 4
initial_interval = 0.05s
max_interval = 0.20s
jitter = 0.00s
```

The simulated transient node fails twice and succeeds on the third attempt.

Observed sequence:

```text
attempt 1 -> failure
wait 0.05s
attempt 2 -> failure
wait 0.10s
attempt 3 -> success
```

The retry verification passed.

## Per-node timeout

Configured node timeout:

```text
0.20s
```

Simulated node duration:

```text
0.60s
```

The node was cancelled cleanly by the timeout rather than hanging.

Observed elapsed time was approximately:

```text
0.21s
```

## Global timeout

Configured global timeout:

```text
0.50s
```

Simulated total duration:

```text
1.00s
```

The simulated graph run was cancelled by the global timeout.

Observed elapsed time was approximately:

```text
0.51s
```

The retry, per-node timeout, and global timeout demonstrations all passed.

Transcript:

```text
transcripts/part4_resilience_demo.txt
```

---

# 19. How to Run the Project

All commands below are run from the project root.

## 19.1 Configure the environment

Use:

```text
MOCK_LLM=true
```

The project does not require an API key for the graded deterministic workflow.

Install the Python dependencies listed in:

```text
requirements.txt
```

---

## 19.2 Run dataset validation

```text
python dataset.py
```

Expected result: deterministic dataset generation and validation with 50 records and a fraud-review rate of 22%.

---

## 19.3 Verify RAG indexing

The RAG/indexing components are under:

```text
rag/
```

The persistent ChromaDB data is stored locally.

The two collections are:

```text
cred_kb_fixed
cred_kb_sentence
```

---

## 19.4 Run Phase 1 evaluation demonstrations

Similarity calibration:

```text
python eval/measure_similarity.py
```

Grounded generation:

```text
python eval/grounded_generation_demo.py
```

Retrieval evaluation:

```text
python eval/retrieval_eval.py
```

The corresponding actual execution transcripts are:

```text
transcripts/part1_dataset_demo.txt
transcripts/part1_indexing_demo.txt
transcripts/part1_similarity_calibration.txt
transcripts/part1_grounded_generation.txt
transcripts/part1_retrieval_eval.txt
```

---

## 19.5 Run the LangGraph agent

The agent interface is:

```text
agent.graph.run_query
```

A normal support query is routed through the complete LangGraph flow, including:

```text
input guardrails
memory loading
intent classification
RAG/status execution
response formatting
structured-output validation
memory persistence
```

---

## 19.6 Run Phase 2 demonstrations

The Phase 2 verification transcripts are:

```text
transcripts/part2_tool_demo.txt
transcripts/part2_routing_demo.txt
transcripts/part2_memory_multiturn.txt
transcripts/part2_memory_fresh.txt
transcripts/part2_guardrails_demo.txt
```

---

## 19.7 Start FastAPI

From the project root:

```text
uvicorn api.main:app --reload
```

The API is available locally.

Endpoints:

```text
GET  /health
POST /ask
```

The verified API demonstrations are preserved in:

```text
transcripts/part3_fastapi_demo.txt
transcripts/part3_logging_demo.txt
```

---

## 19.8 Run the RAG-triad evaluation

Validate the evaluation dataset:

```text
python eval/part3_eval_dataset.py
```

Run the evaluator:

```text
python observability/rag_triad.py
```

The full actual output is preserved in:

```text
transcripts/part3_rag_triad_demo.txt
```

---

## 19.9 Run the MCP demonstration

Start the private local MCP server:

```text
python mcp_server/server.py
```

In a second terminal, run:

```text
python mcp_server/client.py
```

The client connects to:

```text
http://127.0.0.1:8000/mcp
```

and calls:

```text
lookup_loan_application_status
```

for two different record IDs.

Transcript:

```text
transcripts/part4_mcp_demo.txt
```

---

## 19.10 Run the SQLite checkpoint demonstration

```text
python eval/part4_checkpoint.py
```

This creates/uses:

```text
checkpoints.sqlite
```

and demonstrates interruption followed by resumption using the same thread ID.

Transcript:

```text
transcripts/part4_checkpoint_demo.txt
```

---

## 19.11 Run the resilience demonstration

```text
python eval/part4_resilience.py
```

This demonstrates:

* exponential-backoff retry recovery;
* per-node timeout cancellation;
* global graph timeout cancellation.

Transcript:

```text
transcripts/part4_resilience_demo.txt
```

---
# 20. Acceptance Criteria Checklist

The following checklist maps every acceptance criterion in the capstone brief to the implementation and verification evidence.

## Part 1

| Acceptance criterion                                                                                         | Evidence                                                                                              | Status         |
| ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- | -------------- |
| `dataset.py` generates ≥40 records meeting category/status/fraud thresholds and README states design choices | `dataset.py`; `transcripts/part1_dataset_demo.txt`; Dataset Design section above                     | ✅ Demonstrated |
| Knowledge base has ≥12 documents covering every required topic                                               | `knowledge_base/`; `knowledge_base/manifest.json`                                                     | ✅ Demonstrated |
| Both chunking strategies are implemented, embedded and indexed in separate ChromaDB collections              | `rag/chunking.py`; `rag/indexing.py`; `transcripts/part1_indexing_demo.txt`                          | ✅ Demonstrated |
| Grounded generation works on ≥5 in-scope queries plus 1 out-of-scope fallback                                | `rag/generation.py`; `eval/grounded_generation_demo.py`; `transcripts/part1_grounded_generation.txt` | ✅ Demonstrated |
| Precision@3/Recall@3 computed for both collections with per-query arithmetic and recommendation              | `eval/retrieval_eval.py`; `transcripts/part1_retrieval_eval.txt`; retrieval results above            | ✅ Demonstrated |

## Part 2

| Acceptance criterion                                                           | Evidence                                                                                            | Status         |
| ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- | -------------- |
| Status tool correctly looks up records and computes designed escalation score  | `agent/tools.py`; `transcripts/part2_tool_demo.txt`                                                | ✅ Demonstrated |
| LangGraph has ≥4 nodes and conditional routing to both tools                   | `agent/graph.py`; `transcripts/part2_routing_demo.txt`                                             | ✅ Demonstrated |
| Multi-turn memory works and fresh conversation has no inherited state          | `agent/memory.py`; `transcripts/part2_memory_multiturn.txt`; `transcripts/part2_memory_fresh.txt` | ✅ Demonstrated |
| Every response validates against structured-output schema                      | `agent/schema.py`; Phase 2 verification transcripts                                                 | ✅ Demonstrated |
| PII/injection input guardrails and output groundedness guardrail actually fire | `agent/guardrails.py`; `transcripts/part2_guardrails_demo.txt`                                     | ✅ Demonstrated |

## Part 3

| Acceptance criterion                                                                  | Evidence                                                                                                 | Status         |
| ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | -------------- |
| FastAPI backend exposes ≥2 working endpoints with Pydantic models                     | `api/main.py`; `transcripts/part3_fastapi_demo.txt`                                                     | ✅ Demonstrated |
| Every request produces one structured JSONL entry with trace ID                       | `observability/logging.py`; `transcripts/logs/api_requests.jsonl`; `transcripts/part3_logging_demo.txt` | ✅ Demonstrated |
| All three RAG-triad scores reported for all 15 queries plus averages under `MOCK_LLM` | `eval/part3_eval_dataset.py`; `observability/rag_triad.py`; `transcripts/part3_rag_triad_demo.txt`      | ✅ Demonstrated |

## Part 4

| Acceptance criterion                                                                       | Evidence                                                                                                  | Status                                                        |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| Status lookup successfully callable through a real MCP client-server round trip for ≥2 IDs | `mcp_server/server.py`; `mcp_server/client.py`; `transcripts/part4_mcp_demo.txt`                         | ✅ Demonstrated                                                |
| SQLite checkpointing resumes from interruption without re-executing completed nodes        | `eval/part4_checkpoint.py`; `checkpoints.sqlite`; `transcripts/part4_checkpoint_demo.txt` | ✅ Demonstrated in dedicated checkpoint graph |
| Retry recovers transient failure; per-node and global timeouts fire correctly              | `eval/part4_resilience.py`; `transcripts/part4_resilience_demo.txt`                       | ✅ Demonstrated in resilience wrapper         |

---

# 21. Repository Layout

```text
cred-domain-support-agent/
│
├── dataset.py
│
├── knowledge_base/
│   ├── 01_loan_eligibility.md
│   ├── 02_emi_calculation.md
│   ├── 03_credit_card_fees.md
│   ├── 04_kyc_requirements.md
│   ├── 05_fraud_dispute_resolution.md
│   ├── 06_account_closure.md
│   ├── 07_interest_rate_slabs.md
│   ├── 08_prepayment_penalties.md
│   ├── 09_minimum_balance.md
│   ├── 10_credit_score_factors.md
│   ├── 11_joint_account_rules.md
│   ├── 12_nri_account_eligibility.md
│   └── manifest.json
│
├── rag/
│   ├── chunking.py
│   ├── embeddings.py
│   ├── indexing.py
│   ├── retrieval.py
│   └── generation.py
│
├── agent/
│   ├── tools.py
│   ├── graph.py
│   ├── memory.py
│   ├── schema.py
│   └── guardrails.py
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── observability/
│   ├── __init__.py
│   ├── logging.py
│   └── rag_triad.py
│
├── mcp_server/
│   ├── __init__.py
│   ├── server.py
│   └── client.py
│
├── eval/
│   ├── measure_similarity.py
│   ├── grounded_generation_demo.py
│   ├── retrieval_eval.py
│   ├── part3_eval_dataset.py
│   └── ...
│
├── transcripts/
│   ├── memory/
│   ├── logs/
│   ├── part1_dataset_demo.txt
│   ├── part1_indexing_demo.txt
│   ├── part1_similarity_calibration.txt
│   ├── part1_grounded_generation.txt
│   ├── part1_retrieval_eval.txt
│   ├── part2_tool_demo.txt
│   ├── part2_routing_demo.txt
│   ├── part2_memory_multiturn.txt
│   ├── part2_memory_fresh.txt
│   ├── part2_guardrails_demo.txt
│   ├── part3_fastapi_demo.txt
│   ├── part3_logging_demo.txt
│   ├── part3_rag_triad_demo.txt
│   ├── part4_mcp_demo.txt
│   ├── part4_checkpoint_demo.txt
│   └── part4_resilience_demo.txt
│
├── chroma_db/
├── checkpoints.sqlite
├── requirements.txt
├── .env.example
├── .gitignore
├── CONTRACTS.md
└── README.md
```

# 22. Final Completion Summary

The repository's execution evidence is preserved as text transcripts so for actual verification outputs for the individual capstone tasks.

The repository contains the dataset, knowledge base, RAG implementation, LangGraph agent, evaluation, FastAPI deployment, observability, MCP layer, resilience demonstrations, and execution transcripts.

The graded deterministic workflow is based on:

```text
MOCK_LLM=true
```

with no required paid service or hosted LLM API key.


The project implements the full Cred Banking & FinTech capstone across four phases:

```text
Dataset
   ↓
Knowledge Base
   ↓
Dual Chunking + ChromaDB
   ↓
Grounded RAG
   ↓
LangGraph Agent
   ↓
Status Tool + Escalation
   ↓
Persistent Memory
   ↓
Structured Outputs
   ↓
Guardrails
   ↓
FastAPI
   ↓
Structured Observability
   ↓
15-query RAG Triad
   ↓
MCP
   ↓
SQLite Checkpointing
   ↓
Retries + Timeouts
```

