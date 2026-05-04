# Example Run

`example_run.log` is a captured agent session from a real VHH-Screener run.

## What it shows

- **Target**: Human PD-1 at the Pembrolizumab epitope
- **Seed**: Zero-shot (no seed sequence)
- **Iterations**: 6
- **Result**: All 4 tools PASS

Final design metrics:
- Structural liabilities: 0 (7 → 0 over 6 iterations)
- pI: 7.96 (PASS, threshold > 7.5)
- GRAVY: -0.24 (PASS, threshold ≤ 0.0)
- APR max patch: 1.37 (42.5th percentile, PASS, threshold < 95th)

Total cost: $0.0077 (Together AI, DeepSeek V3, ~30k input / 2.6k output tokens)

## How to reproduce

```bash
export TOGETHER_API_KEY=your_key_here
python agent_loop.py --seed none
```

No API key is required to read this log.
