#!/usr/bin/env python3
"""
Convert JSONL file (with optional logits) to labeled test.txt format.
"""

import argparse
import json
import torch


# Standard BIO label mapping
LABEL_MAP = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC",
}


def convert_jsonl_to_labeled(
    jsonl_path: str,
    output_path: str,
    logits_path: str = None,
    use_org_tokens: bool = False,
):
    """Convert JSONL (with optional logits) to labeled test.txt format.

    Args:
        jsonl_path: Path to JSONL file with tokens and ner_tags
        output_path: Path to output labeled file
        logits_path: Optional path to PyTorch logits file (.pt). If provided, use predictions.
        use_org_tokens: If True, use org_tokens and org_ner_tags; if False, use tokens and ner_tags
    """
    # Load JSONL data
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    # Load logits if provided
    logits = None
    if logits_path:
        logits = torch.load(logits_path, map_location="cpu")
        if len(data) != len(logits):
            raise ValueError(
                f"Data size mismatch: {len(data)} examples vs {len(logits)} logit sets"
            )

    # Write output
    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, example in enumerate(data):
            # Get tokens
            if use_org_tokens:
                tokens = example["org_tokens"]
            else:
                tokens = example["tokens"]

            # Get labels
            if logits is not None:
                # Convert logits to predictions
                example_logits = logits[idx]
                predictions = [logit.argmax().item() for logit in example_logits]

                # Verify length matches
                if len(predictions) != len(tokens):
                    raise ValueError(
                        f"Example {idx}: Predictions length ({len(predictions)}) "
                        f"doesn't match tokens length ({len(tokens)})"
                    )

                # Convert to label strings
                labels = [LABEL_MAP.get(pred, f"LABEL_{pred}") for pred in predictions]
            else:
                # Use gold labels from JSONL
                if use_org_tokens:
                    ner_tags = example["org_ner_tags"]
                else:
                    ner_tags = example["ner_tags"]

                # Convert to label strings
                labels = [LABEL_MAP.get(tag, f"LABEL_{tag}") for tag in ner_tags]

            # Write tokens and labels
            for token, label in zip(tokens, labels):
                out_f.write(f"{token} {label}\n")

            # Add blank line to separate sentences
            out_f.write("\n")

    print(f"Converted {len(data)} examples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert JSONL file (with optional logits) to labeled test.txt format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "jsonl_file",
        type=str,
        help="Path to JSONL file with tokens and ner_tags",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to output labeled file (test.txt format)",
    )
    parser.add_argument(
        "--logits-file",
        type=str,
        default=None,
        help="Path to PyTorch logits file (.pt). If provided, use predictions; otherwise use gold labels.",
    )
    parser.add_argument(
        "--use-org-tokens",
        action="store_true",
        help="Use org_tokens and org_ner_tags instead of tokens and ner_tags",
    )

    args = parser.parse_args()

    convert_jsonl_to_labeled(
        args.jsonl_file,
        args.output_file,
        logits_path=args.logits_file,
        use_org_tokens=args.use_org_tokens,
    )
