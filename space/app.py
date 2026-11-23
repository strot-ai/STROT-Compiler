"""STROT live demo (Hugging Face Space).

Tab 1 — Try it: run records through a pre-compiled artifact. No API key, no LLM
call, CPU only. This is STROT's runtime: a frozen pipeline, a 144M local model.

Tab 2 — Compile your own: paste a query + action registry + sample and your own
API key to compile a fresh artifact live. This is STROT's compile step (the one
place an LLM is used).
"""

import os

import gradio as gr
import strot

from demo_tasks import TASKS

STATE = os.environ.get("STROT_STATE", "/app/state")

# Acid theme: neon lime on near-black, monospace.
ACID = gr.themes.Base(
    primary_hue=gr.themes.colors.lime,
    secondary_hue=gr.themes.colors.green,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#0b0f0b",
    body_background_fill_dark="#0b0f0b",
    block_background_fill="#0f140f",
    block_background_fill_dark="#0f140f",
    block_border_color="#1d2a1d",
    body_text_color="#d6ffd6",
    body_text_color_dark="#d6ffd6",
    block_title_text_color="#9fff1a",
    block_label_text_color="#8fdd8f",
    button_primary_background_fill="#9fff1a",
    button_primary_background_fill_hover="#b6ff4d",
    button_primary_text_color="#0b0f0b",
    button_primary_border_color="#9fff1a",
    input_background_fill="#0f140f",
    input_background_fill_dark="#0f140f",
)


def _records(text_block):
    return [{"text": line.strip()} for line in text_block.splitlines() if line.strip()]


def _table(lines, result):
    routed = {o["index"]: o for o in result.outcomes}
    rows = []
    for i, line in enumerate(lines):
        o = routed.get(i)
        if o is None:
            rows.append([line[:90], "— abstained —", ""])
        else:
            rows.append([line[:90], o["action"], f"{o['confidence']:.2f}"])
    return rows


def try_it(task_key, text_block):
    task = TASKS[task_key]
    lines = [l.strip() for l in text_block.splitlines() if l.strip()]
    if not lines:
        return [["(paste at least one record above)", "", ""]]
    try:
        result = strot.run(
            _records(text_block),
            query=task["query"],
            action_registry=task["registry"],
            question=task["question"],
            state=STATE,
            sample_size=min(len(lines), 25),
            measure_labels=True,
        )
    except Exception as exc:  # noqa: BLE001
        return [[
            f"No pre-compiled artifact for this task ({type(exc).__name__}).",
            "Set ANTHROPIC_API_KEY as a build secret and rebuild,",
            "or use the Compile-your-own tab.",
        ]]
    return _table(lines, result)


def compile_your_own(provider, api_key, query, registry_text, records_text):
    import json

    if not api_key.strip():
        return [["Enter your API key above.", "", ""]]
    try:
        registry = json.loads(registry_text)
        assert isinstance(registry, dict) and registry
    except Exception:
        return [["Action registry must be a JSON object: {\"label\": \"description\", ...}", "", ""]]
    lines = [l.strip() for l in records_text.splitlines() if l.strip()]
    if not lines:
        return [["Paste at least one record.", "", ""]]

    env = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    llm = "anthropic/claude-opus-4-8" if provider == "anthropic" else "openai/gpt-4o"
    os.environ[env] = api_key.strip()
    try:
        result = strot.run(
            _records(records_text),
            query=query.strip() or "Classify each record.",
            action_registry=registry,
            llm=llm,
            state="/tmp/byo_state",
            sample_size=min(len(lines), 25),
            measure_labels=True,
        )
    except Exception as exc:  # noqa: BLE001
        return [[f"Compile failed: {type(exc).__name__}", str(exc)[:160], ""]]
    finally:
        os.environ.pop(env, None)
    return _table(lines, result)


INTRO = """
# STROT — the open decision system

**Compile System 2 reasoning into System 1 decisions — $0 at inference.**

**LOW LATENCY · ULTRAFAST · CHEAP & FREE · 0 TOKENS · LOW CARBON · CPU INFERENCE ONLY**

An LLM reasons through your task once, at compile time. Every decision after runs
on a small local model (SupersonicLabs/Julia-1, 144M) on **CPU** — no LLM call,
no API key: faster, cheaper, and a fraction of the energy of per-record LLM calls.
Output is bounded typed-JSON with confidence; low-confidence records abstain.

- **Try it** runs on pre-compiled artifacts: this is the runtime, key-free.
- **Compile your own** uses your API key once to build a fresh artifact.

*Speed: about 60ms/record once the model is warm. This is a free, shared-CPU
Space — the first request after it's been idle wakes the container and loads the
model (about 30–60s), then it's warm.*

Repo: https://github.com/strot-ai/STROT-Compiler · `pip install strot-ai`
"""


def _warmup():
    """Load the model + baked artifacts at boot so the first real request is warm."""
    for task in TASKS.values():
        try:
            strot.run(
                [{"text": task["examples"][0][:1500]}],
                query=task["query"],
                action_registry=task["registry"],
                question=task["question"],
                state=STATE,
                sample_size=1,
                measure_labels=True,
            )
        except Exception:  # noqa: BLE001 - no baked artifact (no build key); skip quietly
            pass


_warmup()

with gr.Blocks(theme=ACID, title="STROT — decision compiler") as demo:
    gr.Markdown(INTRO)

    with gr.Tab("Try it (no key)"):
        task_dd = gr.Dropdown(
            choices=[(t["label"], k) for k, t in TASKS.items()],
            value=next(iter(TASKS)),
            label="Decision task",
        )
        records_in = gr.Textbox(
            value="\n".join(TASKS[next(iter(TASKS))]["examples"]),
            lines=8,
            label="Records (one per line)",
        )
        run_btn = gr.Button("Decide", variant="primary")
        out = gr.Dataframe(headers=["record", "decision", "confidence"], label="Decisions", wrap=True)
        run_btn.click(try_it, [task_dd, records_in], out)

        def _load_examples(task_key):
            return "\n".join(TASKS[task_key]["examples"])

        task_dd.change(_load_examples, task_dd, records_in)

    with gr.Tab("Compile your own (BYO key)"):
        gr.Markdown("Compiles a fresh artifact with your key (used once). Nothing is stored.")
        provider = gr.Radio(["anthropic", "openai"], value="anthropic", label="Synthesizer")
        key_in = gr.Textbox(label="API key", type="password", placeholder="sk-...")
        query_in = gr.Textbox(label="Query (intent)", value="Triage support tickets into the correct workflow.")
        registry_in = gr.Textbox(
            label="Action registry (JSON: label -> description)",
            lines=5,
            value='{\n  "billing": "Refunds, charges, invoices.",\n  "technical": "Bugs, crashes, errors.",\n  "account": "Login, password, access."\n}',
        )
        byo_records = gr.Textbox(lines=6, label="Records (one per line)",
                                 value="I was double charged this month.\nThe app crashes on startup.\nI can't reset my password.")
        byo_btn = gr.Button("Compile & decide", variant="primary")
        byo_out = gr.Dataframe(headers=["record", "decision", "confidence"], label="Decisions", wrap=True)
        byo_btn.click(compile_your_own, [provider, key_in, query_in, registry_in, byo_records], byo_out)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)))
