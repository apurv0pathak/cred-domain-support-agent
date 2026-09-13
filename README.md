# Cred Domain Support Agent

A LangGraph-orchestrated support agent for a fictional lending company, **Cred**.

The agent is designed to answer loan-policy questions from a local knowledge base and look up loan application status from a deterministic dataset.

The project is being developed in phases, with deterministic local execution prioritized before any real LLM/API integrations.

---

## Current Phase

### Phase 1 — Dataset, Knowledge Base & RAG Core

Phase 1 implements:

1. Loan application dataset generation
2. Policy knowledge base
3. Two chunking strategies
4. Local vector indexing with ChromaDB
5. Similarity calibration
6. Grounded answer generation
7. Retrieval evaluation

All Phase 1 components are deterministic and designed to run without API keys.

---

# 1. Loan Application Dataset

The loan application dataset is generated programmatically by `dataset.py`.

### Dataset properties

- Total records: **50**
- Dataset seed: **42**
- Generation attempts: **1**
- Record IDs are deterministic.
- Dataset generation uses Python's seeded random generator.
- Dataset validation is performed before the generated dataset is accepted.

### Record schema

Each loan application contains:

```text
record_id
category
status
loan_amount_inr
days_since_created
flagged_for_fraud_review