"""
agent_loop.py — Sequential Falsification Loop for VHH Design
=============================================================

Implements a Popperian "generate → falsify → critique → mutate" loop using
Together AI (DeepSeek V3) via the OpenAI-compatible API.

The agent acts as a Senior Biologics Lead at Phylo, designing a VHH nanobody
binder for Human PD-1 that targets the same epitope as Pembrolizumab.

Zero-shot binding strategy inspired by the Escalante 180-line approach:
  https://blog.escalante.bio/180-lines-of-code-to-win-the-in-silico-portion-of-the-adaptyv-nipah-binding-competition/

OOD Robustness: The falsifier tools (scan_structural_liabilities,
calculate_biophysical_profile, vhh_hallmark_audit) are deterministic regex /
BioPython checks — not generative — ensuring ground-truth developability
constraints.

Chain of Thought is printed in green and logged to logs/agent_cot.log.

Configuration (env vars):
  TOGETHER_API_KEY  — Required. Your Together AI API key.
  MODEL_ID          — Optional. Defaults to deepseek-ai/DeepSeek-V3.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from openai import OpenAI

# ---------------------------------------------------------------------------
# Import deterministic falsifier tools from the MCP server module
# ---------------------------------------------------------------------------
from biologics_server import (
    calculate_biophysical_profile,
    scan_aggregation_patches,
    scan_structural_liabilities,
    vhh_hallmark_audit,
)

# ---------------------------------------------------------------------------
# Model / provider config
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3"
BASE_URL = "https://api.together.xyz/v1"

# Pricing per million tokens (USD) — update if rates change
PRICE_PER_M_INPUT = 0.20
PRICE_PER_M_OUTPUT = 0.60

# ---------------------------------------------------------------------------
# Logging — all Chain of Thought goes to logs/agent_cot.log AND terminal
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

COT_LOG = LOG_DIR / "agent_cot.log"

cot_logger = logging.getLogger("agent-cot")
cot_logger.setLevel(logging.DEBUG)

_file_handler = logging.FileHandler(COT_LOG)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
)
cot_logger.addHandler(_file_handler)

# ANSI green for terminal CoT
GREEN = "\033[92m"
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
YELLOW = "\033[93m"


def cot_print(msg: str) -> None:
    """Print chain-of-thought in green and log to file."""
    print(f"{GREEN}{msg}{RESET}")
    cot_logger.info(msg)


def header_print(msg: str) -> None:
    """Print a section header in bold cyan."""
    print(f"\n{BOLD}{CYAN}{'=' * 72}")
    print(f"  {msg}")
    print(f"{'=' * 72}{RESET}\n")


def warn_print(msg: str) -> None:
    """Print a warning in yellow."""
    print(f"{YELLOW}{msg}{RESET}")


# ---------------------------------------------------------------------------
# Tool definitions — OpenAI function-calling format
# ---------------------------------------------------------------------------
TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "scan_structural_liabilities",
            "description": (
                "Scan a protein sequence for post-translational modification "
                "hotspots: Deamidation (NG/NS/NA), Isomerization (DG), and "
                "N-glycosylation (N-X-S/T). Returns JSON with liabilities list, "
                "count, and overall PASS/FAIL flag."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "Single-letter amino-acid sequence.",
                    }
                },
                "required": ["sequence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_biophysical_profile",
            "description": (
                "Calculate isoelectric point (pI) and GRAVY hydropathy score "
                "for a protein sequence. Flags aggregation risk: pI < 7.5 = FAIL, "
                "GRAVY > 0.0 = FAIL. Returns structured JSON."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "Single-letter amino-acid sequence.",
                    }
                },
                "required": ["sequence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vhh_hallmark_audit",
            "description": (
                "Audit FR2 hallmark positions (Kabat 37, 44, 45, 47) for "
                "camelid vs. human VH identity. Returns per-position audit "
                "with humanization suggestions and warnings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "Single-letter amino-acid VHH sequence.",
                    },
                    "framework2_start": {
                        "type": "integer",
                        "description": "0-based index where FR2 begins. Default 36.",
                    },
                },
                "required": ["sequence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_aggregation_patches",
            "description": (
                "Scan for aggregation-prone regions (APRs) using clinically-"
                "calibrated sliding-window hydrophobicity. Each 7-residue "
                "window is scored against a reference distribution of 13 "
                "clinical-stage VH/VHH domains. Returns z-scores, percentiles, "
                "Caplacizumab comparison, and PASS/FAIL against the 95th "
                "percentile falsification threshold."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "Single-letter amino-acid sequence.",
                    },
                    "window_size": {
                        "type": "integer",
                        "description": "Sliding window width (default 7).",
                    },
                },
                "required": ["sequence"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Local tool dispatcher — calls the deterministic functions directly
# ---------------------------------------------------------------------------
TOOL_DISPATCH: dict[str, callable] = {
    "scan_structural_liabilities": lambda args: scan_structural_liabilities(
        args["sequence"]
    ),
    "calculate_biophysical_profile": lambda args: calculate_biophysical_profile(
        args["sequence"]
    ),
    "vhh_hallmark_audit": lambda args: vhh_hallmark_audit(
        args["sequence"], args.get("framework2_start", 36)
    ),
    "scan_aggregation_patches": lambda args: scan_aggregation_patches(
        args["sequence"], args.get("window_size", 7)
    ),
}


def execute_tool(name: str, input_args: dict) -> str:
    """Execute a tool by name and return its JSON result string."""
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    return fn(input_args)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a **Senior Biologics Lead** at Phylo, an agentic drug discovery startup.

## Mission
Design a VHH (camelid nanobody) binder for **Human PD-1** that targets the \
same epitope as **Pembrolizumab** (Keytruda). Use a zero-shot binding strategy \
inspired by the Escalante 180-line approach \
(https://blog.escalante.bio/180-lines-of-code-to-win-the-in-silico-portion-of-the-adaptyv-nipah-binding-competition/).

## Sequential Falsification Protocol
You operate under a strict Popperian falsification framework:

1. **Generate**: Propose a full VHH sequence (starting with EVQLV...). The CDR3 \
loop must be designed to mimic the Pembrolizumab heavy-chain CDR3 binding \
geometry against the PD-1 CC' loop / FG loop epitope.

2. **Falsify**: IMMEDIATELY call ALL FOUR tools on your proposed sequence:
   - `vhh_hallmark_audit` — check FR2 hallmark tetrad (positions 37/44/45/47)
   - `scan_structural_liabilities` — check for deamidation (NG/NS/NA), \
isomerization (DG), N-glycosylation (N-X-S/T)
   - `calculate_biophysical_profile` — check pI and GRAVY
   - `scan_aggregation_patches` — sliding-window hydrophobicity scan for \
aggregation-prone regions (sticky patches of 5-7 hydrophobic residues)

3. **Critique**: Analyze every FAIL flag. For each liability found:
   - State the exact motif and position
   - Explain the clinical/manufacturing risk
   - Propose a specific point mutation to resolve it
   - Example: "While this mimics the Pembrolizumab binding loop, the NG motif \
at position X creates a clinical manufacturing risk (asparagine deamidation \
via succinimide intermediate). I am mutating N→Q (or G→A) to eliminate the \
NG sequon."

4. **Mutate & Re-test**: Apply the mutations and re-run ALL FOUR tools on the \
revised sequence. Repeat until all tools return PASS/Low risk.

5. **Final Report**: Once the design passes all checks, present the final \
sequence with a summary of all mutations made and the rationale for each.

## Developability Constraints (hard requirements)
- pI > 7.5 (avoid precipitation near physiological pH)
- GRAVY ≤ 0.0 (hydrophilic surface → lower aggregation)
- No aggregation-prone patches exceeding the 95th percentile of clinical-stage therapeutics
- Zero deamidation motifs (NG, NS, NA) in CDRs
- Zero isomerization motifs (DG) in CDRs
- Zero N-glycosylation sequons (N-X-S/T, X≠Pro) in CDRs
- FR2 hallmark tetrad must be assessed and decision documented

## Output format
Think step by step. Show your reasoning for each design choice. When you \
identify a liability, be specific about position, motif, mechanism, and fix.
"""

# ---------------------------------------------------------------------------
# Main falsification loop
# ---------------------------------------------------------------------------
MAX_ITERATIONS = 10  # Safety cap to prevent runaway loops


PLOT_DIR = Path(__file__).parent / "assets"


def _plot_biophysical_trajectory(
    points: list[dict[str, float | int]],
) -> Path:
    """Generate a pI vs. GRAVY scatter plot showing optimization trajectory.

    Early iterations in red, final iteration in green, with the
    developability safe zone shaded.
    """
    PLOT_DIR.mkdir(exist_ok=True)
    out_path = PLOT_DIR / "biophysical_trajectory.png"

    pis = [p["pI"] for p in points]
    gravys = [p["gravy"] for p in points]
    iters = [p["iteration"] for p in points]

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0a0a0a")
    ax.set_facecolor("#0a0a0a")

    # Safe zone: pI > 7.5 and GRAVY <= 0.0
    ax.axhspan(-0.8, 0.0, xmin=0, xmax=1, alpha=0.12, color="#00ff41", zorder=0)
    ax.axhline(0.0, color="#00ff41", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(7.5, color="#00ff41", linewidth=0.8, linestyle="--", alpha=0.5)

    # Label the safe zone
    ax.text(
        max(max(pis), 8.5) + 0.1,
        -0.4,
        "SAFE\nZONE",
        color="#00ff41",
        fontsize=9,
        alpha=0.4,
        ha="left",
        va="center",
        fontfamily="monospace",
    )

    # Draw trajectory line connecting points in iteration order
    ax.plot(pis, gravys, color="#444444", linewidth=1, zorder=1)

    # Color: first point red, last point green, middle points gradient
    n = len(points)
    for i, (pi, gravy, it) in enumerate(zip(pis, gravys, iters)):
        if i == 0:
            color = "#ff3333"
            label = f"Iter {it} (initial)"
            size = 80
        elif i == n - 1:
            color = "#00ff41"
            label = f"Iter {it} (final)"
            size = 100
        else:
            # Gradient from red to yellow to green
            frac = i / (n - 1)
            color = plt.cm.RdYlGn(frac)
            label = f"Iter {it}"
            size = 50

        ax.scatter(
            pi,
            gravy,
            c=[color],
            s=size,
            zorder=2,
            edgecolors="white",
            linewidths=0.5,
            label=label,
        )

    ax.set_xlabel(
        "Isoelectric Point (pI)", color="white", fontsize=11, fontfamily="monospace"
    )
    ax.set_ylabel("GRAVY Score", color="white", fontsize=11, fontfamily="monospace")
    ax.set_title(
        "VHH-Falsifier: Biophysical Optimization Trajectory",
        color="white",
        fontsize=13,
        fontfamily="monospace",
        pad=15,
    )

    # Style axes
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#333333")

    # Threshold annotations
    ax.annotate(
        "pI = 7.5",
        xy=(7.5, ax.get_ylim()[1]),
        fontsize=8,
        color="#00ff41",
        alpha=0.6,
        fontfamily="monospace",
        ha="center",
        va="bottom",
    )
    ax.annotate(
        "GRAVY = 0.0",
        xy=(ax.get_xlim()[0], 0.0),
        fontsize=8,
        color="#00ff41",
        alpha=0.6,
        fontfamily="monospace",
        ha="left",
        va="bottom",
    )

    ax.legend(
        loc="upper left",
        fontsize=8,
        facecolor="#1a1a1a",
        edgecolor="#333333",
        labelcolor="white",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="#0a0a0a")
    plt.close(fig)

    subprocess.run(["open", str(out_path)], check=False)

    return out_path


def run_falsification_loop() -> None:
    """Run the generate → falsify → critique → mutate loop."""
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        print(
            "ERROR: Set TOGETHER_API_KEY environment variable before running.\n"
            "  Get one at https://api.together.xyz/settings/api-keys",
            file=sys.stderr,
        )
        sys.exit(1)

    model_id = os.environ.get("MODEL_ID", DEFAULT_MODEL)

    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    header_print("VHH-Falsifier — Sequential Falsification Loop")
    cot_print(f"Session started: {datetime.now(timezone.utc).isoformat()}")
    cot_print("Target: Human PD-1 (Pembrolizumab epitope)")
    cot_print("Scaffold: Camelid VHH nanobody")
    cot_print(f"Provider: Together AI ({BASE_URL})")
    cot_print(f"Model: {model_id}")
    cot_print(f"Max iterations: {MAX_ITERATIONS}")
    cot_print(f"CoT log: {COT_LOG.resolve()}\n")

    # Initial user message kicks off the loop
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Design a VHH nanobody targeting Human PD-1 at the "
                "Pembrolizumab epitope. Follow the sequential falsification "
                "protocol exactly. Begin by proposing your first candidate "
                "sequence, then immediately falsify it with all three tools."
            ),
        },
    ]

    # Cost tracking
    total_input_tokens = 0
    total_output_tokens = 0

    # Biophysical trajectory tracking for scatter plot
    biophysical_points: list[dict[str, float | int]] = []

    iteration = 0
    for iteration in range(1, MAX_ITERATIONS + 1):
        header_print(f"ITERATION {iteration}")

        # Call model with tools
        response = client.chat.completions.create(
            model=model_id,
            max_tokens=4096,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        choice = response.choices[0]
        finish_reason = choice.finish_reason
        message = choice.message

        # Track token usage and cost
        if response.usage:
            iter_in = response.usage.prompt_tokens
            iter_out = response.usage.completion_tokens
            total_input_tokens += iter_in
            total_output_tokens += iter_out
            iter_cost = (
                iter_in * PRICE_PER_M_INPUT / 1_000_000
                + iter_out * PRICE_PER_M_OUTPUT / 1_000_000
            )
            running_cost = (
                total_input_tokens * PRICE_PER_M_INPUT / 1_000_000
                + total_output_tokens * PRICE_PER_M_OUTPUT / 1_000_000
            )
            cot_print(
                f"[Cost] Iteration: {iter_in:,} in / {iter_out:,} out "
                f"= ${iter_cost:.4f}  |  Running total: ${running_cost:.4f}"
            )

        cot_print(f"[Iteration {iteration}] Finish reason: {finish_reason}")

        # Print text reasoning in green
        if message.content:
            cot_print(f"\n[Agent CoT — Iteration {iteration}]")
            for line in message.content.splitlines():
                cot_print(f"  {line}")

        # Append assistant message to history
        messages.append(message.model_dump(exclude_none=True))

        # If no tool calls, the agent has reached a conclusion
        if finish_reason == "stop" or not message.tool_calls:
            header_print("FALSIFICATION LOOP COMPLETE")
            cot_print("Agent reached final conclusion.")
            break

        # Execute each tool call and feed results back
        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            cot_print(
                f"\n[Tool Call] {fn_name}({json.dumps(fn_args, indent=None)[:120]}...)"
            )

            result_str = execute_tool(fn_name, fn_args)
            result_data = json.loads(result_str)

            cot_print(f"[Tool Result] {fn_name}:")
            cot_print(f"  {json.dumps(result_data, indent=2)[:500]}")

            # Capture pI/GRAVY for trajectory plot
            if (
                fn_name == "calculate_biophysical_profile"
                and "error" not in result_data
            ):
                biophysical_points.append(
                    {
                        "iteration": iteration,
                        "pI": result_data["isoelectric_point"],
                        "gravy": result_data["gravy"],
                    }
                )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str,
                }
            )

    else:
        warn_print(
            f"\nWARNING: Reached max iterations ({MAX_ITERATIONS}) "
            "without converging on a passing design."
        )

    # Final cost summary
    final_cost = (
        total_input_tokens * PRICE_PER_M_INPUT / 1_000_000
        + total_output_tokens * PRICE_PER_M_OUTPUT / 1_000_000
    )
    header_print("COST SUMMARY")
    cot_print(f"Input tokens:  {total_input_tokens:,}")
    cot_print(f"Output tokens: {total_output_tokens:,}")
    cot_print(f"Total cost:    ${final_cost:.4f}")

    # Generate biophysical trajectory plot
    if len(biophysical_points) >= 2:
        plot_path = _plot_biophysical_trajectory(biophysical_points)
        cot_print(f"Biophysical trajectory plot saved: {plot_path}")
    elif biophysical_points:
        cot_print("Only one biophysical data point — skipping plot.")
    else:
        cot_print("No biophysical data captured — skipping plot.")

    # Write session summary to log
    cot_print(f"\nSession ended: {datetime.now(timezone.utc).isoformat()}")
    cot_print(f"Total iterations: {iteration}")
    cot_print(f"Full CoT log: {COT_LOG.resolve()}")


if __name__ == "__main__":
    run_falsification_loop()
