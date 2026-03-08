# VHH-Falsifier: Agentic Sequential Falsification for Nanobody Engineering

> *"The criterion of the scientific status of a theory is its falsifiability."*
> — Karl Popper, *The Logic of Scientific Discovery* (1959)

## Overview

VHH-Falsifier is an agentic computational biologics platform that applies Popperian sequential falsification to the engineering of camelid VHH nanobodies. Unlike conventional generative design pipelines that propose candidates and hope they survive downstream triage, VHH-Falsifier inverts the paradigm: the agent aggressively attempts to disprove the manufacturability of its own designs at every iteration, converging only when a candidate withstands all falsification pressure.

The system implements a closed-loop architecture in which an LLM-driven "Generator" proposes VHH sequences and a suite of deterministic "Falsifier" tools—grounded in regex pattern matching and BioPython analytics, not generative inference—systematically stress-test each candidate against clinical and manufacturing developability constraints. No design exits the loop unfalsified.

This is a component of the [Biomni Lab](https://github.com/ChristopherSNelson) ecosystem for agentic drug discovery.

## Design Philosophy

Traditional *in silico* antibody design operates under a verification bias: generate a candidate, confirm it looks reasonable, advance it. This systematically underweights liabilities that surface late in development—deamidation hotspots that degrade shelf life, aggregation-prone hydrophobic patches, glycosylation sequons that compromise batch consistency.

VHH-Falsifier inverts this with a sequential falsification protocol:

```
GENERATE → FALSIFY → CRITIQUE → MUTATE → RE-FALSIFY → ... → PASS
```

1. Generate — The agent proposes a full VHH sequence with CDR loops designed for target engagement.
2. Falsify — Every candidate is immediately subjected to all deterministic developability checks. The agent does not proceed until tools return results.
3. Critique — Each failure is diagnosed with mechanistic specificity: the exact motif, its position, the biochemical liability, and the clinical consequence.
4. Mutate — Point mutations are proposed to eliminate each liability while preserving binding geometry.
5. Re-Falsify — The revised candidate is re-tested from scratch. The loop repeats until all checks pass or the iteration budget is exhausted.

## Technical Heritage

The zero-shot binding strategy draws on the [Escalante 180-line approach](https://blog.escalante.bio/180-lines-of-code-to-win-the-in-silico-portion-of-the-adaptyv-nipah-binding-competition/) to computational antibody design, adapted here for the VHH scaffold and extended with a falsification layer that the original strategy does not include. Where Escalante demonstrated that a compact codebase can produce competitive binders *in silico*, VHH-Falsifier adds the developability gauntlet that separates a binding prediction from a manufacturable therapeutic candidate.

## Key Features

### Deterministic Liability Scanning (PTM Hotspots)

Regex-based detection of post-translational modification motifs known to cause manufacturing failures in biologics:

| Liability | Motif | Mechanism |
|---|---|---|
| Deamidation | NG, NS, NA | Asparagine deamidation via succinimide intermediate |
| Isomerization | DG | Aspartate isomerization to iso-Asp |
| N-Glycosylation | N-X-S/T (X != P) | Aberrant glycosylation at consensus sequons |

All scanning is deterministic—no LLM inference, no stochastic variation. Ground truth for falsification.

### Biophysical Stress-Testing (pI / GRAVY)

Quantitative assessment of aggregation propensity via two orthogonal metrics:

- Isoelectric point (pI) — Candidates with pI < 7.5 are flagged for precipitation risk near physiological pH. Computed via BioPython's `ProteinAnalysis`.
- GRAVY hydropathy score — Candidates with GRAVY > 0.0 are flagged for elevated hydrophobicity and aggregation propensity (Kyte & Doolittle, 1982).

### VHH Hallmark Auditing (FR2 Tetrad)

Structural integrity check of the four FR2 positions (Kabat 37, 44, 45, 47) that distinguish camelid VHH domains from conventional human VH:

| Kabat Position | Camelid Residue | Human VH Residue | Role |
|---|---|---|---|
| 37 | F | V | Core packing; compensates for missing VL |
| 44 | E | G | Hydrophilic substitution at former VH-VL interface |
| 45 | R | L | Charged residue replacing hydrophobic VL contact |
| 47 | G | W | Flexible Gly replacing bulky Trp |

The audit flags humanization decisions that may destabilize the VHH scaffold and documents the rationale for each retained or substituted hallmark residue.

## Architecture

```
agent_loop.py                    biologics_server.py
┌─────────────────────┐          ┌──────────────────────────────┐
│  LLM Agent          │          │  FastMCP Server              │
│  (DeepSeek V3 /     │  tools   │                              │
│   Together AI)      │────────→ │  scan_structural_liabilities │
│                     │          │  calculate_biophysical_profile│
│  Generate → Falsify │←────────│  vhh_hallmark_audit          │
│  → Critique → Mutate│  JSON    │                              │
└─────────────────────┘          └──────────────────────────────┘
        │
        ▼
  logs/agent_cot.log
  (full Chain of Thought)
```

- `biologics_server.py` — FastMCP server exposing three deterministic falsification tools over stdio. Each tool returns structured JSON designed to be agent-readable.
- `agent_loop.py` — Orchestrates the sequential falsification loop via the OpenAI-compatible API (Together AI / DeepSeek V3). Includes per-iteration cost tracking. All Chain of Thought is printed in green and logged to `logs/agent_cot.log`.

## Quickstart

```bash
# Clone
git clone https://github.com/ChristopherSNelson/VHH-Falsifier.git
cd VHH-Falsifier

# Install dependencies
pip install fastmcp biopython openai

# Set API key (Together AI)
export TOGETHER_API_KEY="your-key-here"

# Run the falsification loop
python agent_loop.py
```

### Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `TOGETHER_API_KEY` | *(required)* | Together AI API key |
| `MODEL_ID` | `deepseek-ai/DeepSeek-V3` | Model identifier (any OpenAI-compatible model on Together) |

## Output

The agent prints its Chain of Thought in green terminal output, with section headers in cyan. A cost summary is printed at the end of each run:

```
========================================================================
  COST SUMMARY
========================================================================

Input tokens:  8,432
Output tokens: 3,210
Total cost:    $0.0036
```

All reasoning and tool results are persisted to `logs/agent_cot.log` for post-hoc analysis and reproducibility.

## Developability Constraints

These are hard requirements enforced by the falsification loop. A candidate cannot pass unless all constraints are satisfied:

| Constraint | Threshold | Rationale |
|---|---|---|
| Isoelectric point | pI > 7.5 | Avoid precipitation near physiological pH |
| Hydropathy | GRAVY <= 0.0 | Minimize aggregation propensity |
| Deamidation motifs | Zero in CDRs | Eliminate shelf-life degradation risk |
| Isomerization motifs | Zero in CDRs | Prevent charge heterogeneity |
| N-Glycosylation sequons | Zero in CDRs | Ensure batch consistency |
| FR2 hallmark tetrad | Assessed and documented | Structural integrity of VHH scaffold |

## License

MIT

## Author

Chris Nelson

- [LinkedIn](https://www.linkedin.com/in/christopher-s-nelson/)
- [GitHub](https://github.com/ChristopherSNelson)
