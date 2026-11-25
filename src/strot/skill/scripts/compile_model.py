#!/usr/bin/env python3
"""Compile a STROT decision model from agent-supplied labels — the LLM-free door.

The agent designs the typed questions and labels a small sample itself, writes
them to JSON, and runs this. STROT calls no LLM and holds no key; it embeds the
records, fits a calibrated head per question plus an out-of-distribution gate,
and bakes a self-contained ``<name>.strot`` model that decides every record after
at $0.

    python compile_model.py --data data.json --questions questions.json \
        --labels labels.json --name triage [--eval-labels held_out.json]

For images, each record carries an image reference (a path or URL) under some
field; embed it with a frozen vision encoder (head-tier only):

    python compile_model.py --data data.json --questions questions.json \
        --labels labels.json --name defect --modality image --image-field photo

Shapes:
  data.json      [ {record}, ... ]   # image: record[<image-field>] is a path/URL
  questions.json { "<name>": {"type": "predicate|choice|score", ...}, ... }
  labels.json    [ {"<question>": <answer>}, ... ]   # one per record, same order
  held_out.json  [ {"record": {...}, "expect": {"<question>": <answer>}}, ... ]

Needs the compile extra: pip install "strot-ai[local]".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compile a STROT decision model from labels.")
    ap.add_argument("--data", required=True, help="JSON list of records")
    ap.add_argument("--questions", required=True, help="JSON typed-question config")
    ap.add_argument("--labels", required=True,
                    help="JSON list of {question: answer}, one per record")
    ap.add_argument("--name", required=True, help="output model name -> <name>.strot")
    ap.add_argument("--mode", default="head", choices=["head", "deep"],
                    help="head (default) or deep (LoRA; needs [local])")
    ap.add_argument("--projection", default=None,
                    help="optional path to a route_context(record) python source file")
    ap.add_argument("--modality", default="text", choices=["text", "image"],
                    help="text (default) or image (embed record[image-field] with a vision encoder; head-tier only)")
    ap.add_argument("--encoder", default=None,
                    help="override the base encoder (image: a SigLIP/ViT/DINOv2 checkpoint)")
    ap.add_argument("--image-field", default=None,
                    help="with --modality image: the record key holding the image (path/URL)")
    ap.add_argument("--eval-labels", default=None,
                    help="optional JSON [{record, expect}] to score the model")
    args = ap.parse_args(argv)

    try:
        import strot
    except ImportError:
        print("strot is not installed. Run: pip install \"strot-ai[local]\"", file=sys.stderr)
        return 2

    data = _load(args.data)
    questions = _load(args.questions)
    labels = _load(args.labels)
    projection = Path(args.projection).read_text(encoding="utf-8") if args.projection else None

    if len(labels) != len(data):
        print(f"error: {len(labels)} labels for {len(data)} records — they must match "
              "one-to-one, in the same order.", file=sys.stderr)
        return 2

    result = strot.compile(
        data, questions, labels, projection=projection, mode=args.mode,
        name=args.name, execute=False, modality=args.modality,
        encoder=args.encoder, image_field=args.image_field,
    )
    model_path = result.model.dispatch(args.name)
    print(f"compiled: {model_path}")

    if args.eval_labels:
        model = strot.load_model(str(model_path))
        report = strot.evaluate(model, _load(args.eval_labels))
        print("eval:", json.dumps(report.get("overall", report), indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
