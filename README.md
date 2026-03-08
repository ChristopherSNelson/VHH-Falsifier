# VHH-Falsifier: Agentic Sequential Falsification for Nanobody Engineering

## What This Does

An LLM agent designs VHH nanobody sequences, then immediately tries to break them. Deterministic tools scan for manufacturing liabilities (deamidation, aggregation, glycosylation). If anything fails, the agent mutates the design and re-tests. The loop repeats until the candidate passes every check — or the iteration budget runs out.

No design exits the loop unfalsified.

Part of the [Biomni Lab](https://github.com/ChristopherSNelson) ecosystem for agentic drug discovery.

## The Loop

```
GENERATE → FALSIFY → CRITIQUE → MUTATE → RE-FALSIFY → ... → PASS
```

1. Generate — Propose a VHH sequence with CDR loops targeting the binding epitope.
2. Falsify — Run all three deterministic tools against the candidate.
3. Critique — Diagnose each failure: exact motif, position, mechanism, clinical consequence.
4. Mutate — Apply point mutations to fix liabilities while preserving binding geometry.
5. Re-Falsify — Re-test from scratch. Repeat until clean.

## Technical Heritage

Zero-shot binding strategy adapted from the [Escalante 180-line approach](https://blog.escalante.bio/180-lines-of-code-to-win-the-in-silico-portion-of-the-adaptyv-nipah-binding-competition/), extended with a developability falsification layer.

## Falsification Tools

### Liability Scanning (PTM Hotspots)

Deterministic regex — no LLM inference, no stochastic variation.

| Liability | Motif | Mechanism |
|---|---|---|
| Deamidation | NG, NS, NA | Asparagine deamidation via succinimide intermediate |
| Isomerization | DG | Aspartate isomerization to iso-Asp |
| N-Glycosylation | N-X-S/T (X != P) | Aberrant glycosylation at consensus sequons |

### Biophysical Profiling (pI / GRAVY)

- pI < 7.5 → precipitation risk near physiological pH
- GRAVY > 0.0 → elevated hydrophobicity, aggregation-prone

### VHH Hallmark Audit (FR2 Tetrad)

Checks Kabat positions 37, 44, 45, 47 for camelid vs. human VH identity:

| Kabat Position | Camelid | Human VH | Role |
|---|---|---|---|
| 37 | F | V | Core packing; compensates for missing VL |
| 44 | E | G | Hydrophilic substitution at former VH-VL interface |
| 45 | R | L | Charged residue replacing hydrophobic VL contact |
| 47 | G | W | Flexible Gly replacing bulky Trp |

## Architecture

```
agent_loop.py                    biologics_server.py
┌─────────────────────┐          ┌──────────────────────────────┐
│  LLM Agent          │          │  FastMCP Server              │
│  (DeepSeek V3 /     │  tools   │                              │
│   Together AI)      │────────→ │ scan_structural_liabilities  │
│                     │          │ calculate_biophysical_profile│
│  Generate → Falsify │←──────── │ vhh_hallmark_audit           │
│  → Critique → Mutate│  JSON    │                              │
└─────────────────────┘          └──────────────────────────────┘
        │
        ▼
  logs/agent_cot.log
```

- `biologics_server.py` — FastMCP server, three deterministic tools, structured JSON output.
- `agent_loop.py` — Falsification loop via OpenAI-compatible API. Per-iteration cost tracking. Green CoT terminal output, logged to `logs/agent_cot.log`.

## Quickstart

```bash
git clone https://github.com/ChristopherSNelson/VHH-Falsifier.git
cd VHH-Falsifier
pip install fastmcp biopython openai
export TOGETHER_API_KEY="your-key-here"
python agent_loop.py
```

| Environment Variable | Default | Description |
|---|---|---|
| `TOGETHER_API_KEY` | *(required)* | Together AI API key |
| `MODEL_ID` | `deepseek-ai/DeepSeek-V3` | Any OpenAI-compatible model on Together |

## Developability Constraints

Hard requirements. Nothing passes unless all are satisfied.

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
