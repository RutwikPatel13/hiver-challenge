
# Build an AI Email Suggested-Response System

## Overview

Build a system that, given an incoming email, generates a suggested reply — learning from a dataset of past emails and their responses. Then build a way to measure how good each generated response actually is.

## Task Requirements

### 1. Build a Dataset

- **Objective**: Create (or source) a dataset of past emails paired with the replies that were sent
- **Options**: Synthetic, from public corpora, or hand-authored — your choice
- **Deliverable**: Dataset with clear documentation
- **Documentation**: Explain in README where it came from and why it's representative

### 2. Generate Suggested Responses (Gen AI)

- **Objective**: Build a system that takes a new incoming email and produces a suggested reply using a generative AI model (an LLM)
- **Requirement**: Not a classical ML classifier — use Gen AI
- **Grounding**: Ground the generation in your dataset
- **Approaches**: Choose how — prompting, RAG/retrieval over past emails, few-shot examples, fine-tuning an LLM, or a mix
- **Deliverable**: Justify the trade-offs in your README

### 3. Measure Accuracy — The Core Challenge

This is what we care about most. Build an accuracy system that, for a generated response, tells us how accurate/good it is and why.

**Key Considerations**:

- What "accurate" even means for a suggested reply (exact match is too strict)
- The metric(s) you use and why they're the right ones
- How you validate the metric reflects real quality, not just a number
- Reporting: per-response scores and an overall system score

## What We're Evaluating

1. **Clarity of thinking about accuracy/evaluation** (weighted heaviest)
2. **Quality and honesty of the dataset**
3. **Whether the response generator is sensible and runs**
4. **README covering your approach, trade-offs, and how to run it**
5. **Ship something that runs end-to-end**
6. **How you used AI tools** (document in README)

## Deliverables

- [ ] A public GitHub repository URL
- [ ] The dataset (or a script that generates/fetches it) with documentation on how you built it
- [ ] The Gen-AI response generator, runnable end-to-end
- [ ] The accuracy/evaluation system, with per-response and overall scores
- [ ] A README covering:
  - Your approach
  - Why your accuracy metric is right
  - How to run it
  - How you used AI tools

## Submission

- Open the submission form and enter your public GitHub repo link
- You'll get a confirmation from Notion

**Submission Form**: Challenge Submissions

- GitHub URL: [TO BE FILLED]
- Email: rutwikpatel1313@gmail.com
