import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    load_data,
    load_alignments,
    build_alignment_mapping,
    extract_entity_indices,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def map_entities(
    from_indices: List[int],
    fromid2toid: Dict[int, List[int]],
    complete_from: bool,  # From of the mapping
    complete_to: bool,  # To of the mapping
) -> Optional[List[int]]:
    """Map from entities to to entities using alignment.

    Args:
        from_indices: Indices of the from entity
        fromid2toid: From-to-to alignment mapping
        complete_from: Checks whether the from mapping is complete
        complete_to: Checks whether the to mapping is complete

    Returns:
        List of to indices
    """

    to_indices = []
    # Collect the corresponding to indices
    for from_idx in from_indices:
        if from_idx in fromid2toid:
            to_indices = to_indices + fromid2toid[from_idx]
        elif complete_from:  # COMP-FROM
            # We couldn´t map the from mapping entity completely
            return None

    if not to_indices:
        return None  # Nothing to map

    # Build the span of indices
    to_indices = sorted(set(to_indices))
    min_idx, max_idx = to_indices[0], to_indices[-1]

    # Check that the entity covers a consecutive number of indices (COMP-TO)
    if complete_to:  # COMP-TO
        expected_indices = list(range(min_idx, max_idx + 1))
        if to_indices != expected_indices:
            # The to mapping entity is incomplete
            return None

    return to_indices


def project_translate_test_labels_bio(
    from_data_path,
    to_data_path,
    alignment_path,
    to_output_path,
    complete_from=False,
    complete_to=False,
    restrict_to=False,
    **kwargs,
):
    logger.info(f"Loading labeled from data from {from_data_path}")
    from_data = load_data(from_data_path, with_labels=True)

    logger.info(f"Loading unlabeled to data from {to_data_path}")
    to_data = load_data(to_data_path, with_labels=False)

    logger.info(f"Loading alignments from {alignment_path}")
    alignments = load_alignments(alignment_path)

    # Sanity check: As many examples in all datasets
    if len(from_data) != len(to_data) or len(from_data) != len(alignments):
        raise ValueError(
            f"Data size mismatch: {len(from_data)} from examples vs "
            f"{len(to_data)} to examples vs {len(alignments)} alignments"
        )

    # Projecting labels
    total_entities = 0
    all_to_labels = []

    for (from_tokens, from_labels), to_tokens, alignment in zip(
        from_data, to_data, alignments
    ):
        # Sanity check: as many labels as tokens in from data
        if len(from_tokens) != len(from_labels):
            raise ValueError(
                f"From tokens length ({len(from_tokens)}) doesn't match labels length ({len(from_labels)})"
            )

        # Create place holder for projected labels
        to_labels = ["O"] * len(to_tokens)

        # Create mapping lookup from from token id to list of to token ids
        # alignment is already a list of [from, to] pairs
        fromid2toid = build_alignment_mapping(alignment, inverse=True)

        # Get the entity´s indices of the from labels
        # From: ['O','O','B-PER','I-PER','O',...]
        # To: [[2,3],...]
        from_entities, entity_labels = extract_entity_indices(from_labels)

        # Sanity check that we found all entities
        flattened_entities = sum(from_entities, [])
        reconstructed = [
            from_labels[idx] if idx in flattened_entities else "O"
            for idx in range(len(from_labels))
        ]
        if from_labels != reconstructed:
            logger.warning(
                "Entity extraction validation failed - some entities may be missed"
            )

        # Project each entity (labels from from data)
        for entity_indices, entity_label in zip(from_entities, entity_labels):

            to_indices = map_entities(
                entity_indices,
                fromid2toid,
                complete_from,
                complete_to,
            )

            if to_indices is None:
                continue

            min_idx, max_idx = min(to_indices), max(to_indices)

            # Set the first token with B- tag
            to_labels[min_idx] = from_labels[entity_indices[0]]

            # Check the filters
            if restrict_to:
                if len(entity_indices) == 1:
                    continue

            # We have multiple to mappings
            if min_idx != max_idx:
                if len(entity_indices) == 1:
                    # We have a single from token, but multiple to tokens
                    current_tag = from_labels[entity_indices[0]]
                    if current_tag.startswith("B-"):
                        # We have a B-Tag at the beginning, use I-Tag for rest
                        i_tag = "I-" + current_tag[2:]
                    else:
                        # We have an I-Tag at the beginning, use it
                        i_tag = current_tag
                else:
                    # We have multiple from tokens and multiple to tokens, take the last label
                    i_tag = from_labels[entity_indices[-1]]
                    # Make sure it's an I-tag
                    if i_tag.startswith("B-"):
                        i_tag = "I-" + i_tag[2:]

                for jj in range(min_idx + 1, max_idx + 1):
                    to_labels[jj] = i_tag

        # Sanity check
        if len(to_labels) != len(to_tokens):
            raise ValueError("Length of projected labels doesn't match to tokens")

        all_to_labels.append((to_tokens, to_labels))

        # Count projected entities
        entities, _ = extract_entity_indices(to_labels)
        total_entities += len(entities)

    # Sanity check
    if len(all_to_labels) != len(to_data):
        raise ValueError("Number of projected instances doesn't match to data")

    logger.info(f"Projected {total_entities} entities total")

    # Save results
    logger.info(f"Saving projected labels to {to_output_path}")
    with open(to_output_path, "w", encoding="utf-8") as f:
        for tokens, labels in all_to_labels:
            for token, label in zip(tokens, labels):
                f.write(f"{token} {label}\n")
            f.write("\n")
    logger.info("Projection complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "from_data_path",
        help="Path to the labeled from data (source language with labels)",
    )
    parser.add_argument(
        "to_data_path",
        help="Path to the unlabeled to data (target language without labels)",
    )
    parser.add_argument(
        "alignment_path",
        help="Path to the alignment file",
    )
    parser.add_argument(
        "to_output_path",
        help="Path to save the projected labeled to data",
    )
    parser.add_argument(
        "--complete_from",
        action="store_true",
        help="Require complete from entity alignment",
    )
    parser.add_argument(
        "--complete_to",
        action="store_true",
        help="Require complete to entity alignment",
    )
    parser.add_argument(
        "--restrict_to",
        action="store_true",
        help="Restricted to entity alignment",
    )
    args = parser.parse_args()

    project_translate_test_labels_bio(
        args.from_data_path,
        args.to_data_path,
        args.alignment_path,
        args.to_output_path,
        complete_from=args.complete_from,
        complete_to=args.complete_to,
        restrict_to=args.restrict_to,
    )
