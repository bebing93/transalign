"""
Common utility functions for NER alignment and evaluation tasks.
"""

from typing import List, Dict, Tuple
from collections import defaultdict


def load_data(file_path: str, with_labels: bool = False):
    """Load data from a file like test.txt.

    Args:
        file_path: Path to the file
        with_labels: If True, expect and load labels; if False, load only tokens

    Returns:
        If with_labels=True: List of (tokens, labels) tuples
        If with_labels=False: List of token lists
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        tokens = []
        labels = []
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    if with_labels:
                        data.append((tokens, labels))
                    else:
                        data.append(tokens)
                    tokens = []
                    labels = []
            else:
                parts = line.split()
                if parts:
                    tokens.append(parts[0])
                    if with_labels and len(parts) >= 2:
                        labels.append(parts[1])
        if tokens:
            if with_labels:
                data.append((tokens, labels))
            else:
                data.append(tokens)
    return data


def load_alignments(file_path: str) -> List[List[List[int]]]:
    """Load alignments from a file.

    Args:
        file_path: Path to the alignment file

    Returns:
        List of alignment lists (each alignment is a list of [from_idx, to_idx] pairs)
    """
    alignments = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            parts = line.strip().split("\t")
            if len(parts) < 2:
                raise ValueError(
                    f"Line {line_num} in {file_path}: Missing ID or alignment. "
                    f"Expected format: 'ID\\talignment1 alignment2 ...'"
                )
            # parts[0] is the ID, parts[1] is the alignment string
            alignment_str = parts[1].strip()
            alignment_pairs = []
            if alignment_str:
                # Split alignment string into individual alignments and convert to pairs
                for align in alignment_str.split():
                    align_parts = align.split("-")
                    if len(align_parts) == 2:
                        alignment_pairs.append(
                            [int(align_parts[0]), int(align_parts[1])]
                        )
            alignments.append(alignment_pairs)
    return alignments


def build_alignment_mapping(
    alignment_line: List[List[int]], inverse=False
) -> Dict[int, List[int]]:
    """Build source-to-target alignment mapping.

    Args:
        alignment_line: List of [source_idx, target_idx] pairs
        inverse: If true, maps from target idx to source idx else vice-versa

    Returns:
        Dictionary mapping source indices to sorted target indices
    """
    from2to = defaultdict(list)
    for src_idx, trg_idx in alignment_line:
        if inverse:
            from2to[trg_idx].append(src_idx)
        else:
            from2to[src_idx].append(trg_idx)

    return {k: sorted(v) for k, v in from2to.items()}


def extract_entity_indices(labels: List[str]) -> Tuple[List[List[int]], List[str]]:
    """Extract entity indices and types from BIO labels.

    Args:
        labels: List of BIO tag strings (e.g., 'O', 'B-PER', 'I-PER')

    Returns:
        Tuple of (entity_indices, entity_types)
        - entity_indices: List of lists, where each inner list contains token indices for an entity
        - entity_types: List of entity labels (the B- tag for each entity)
    """
    entity = []
    entities = []
    entity_types = []
    in_entity = False

    for j, tag in enumerate(labels):
        if tag == "O":  # O Tag
            if in_entity:
                # End of previous entity reached
                entities.append(entity)
                entity_types.append(labels[entity[0]])
                in_entity = False
            entity = []
            continue
        elif tag.startswith("B-"):  # B-Tag
            if in_entity:
                # End of previous entity reached
                entities.append(entity)
                entity_types.append(labels[entity[0]])

            # Start new entity
            in_entity = True
            entity = [j]
        elif tag.startswith("I-"):  # I-Tag
            if in_entity:
                # Within current entity
                entity.append(j)
            else:
                # Orphaned I tag - treat as beginning of new entity
                in_entity = True
                entity = [j]

    # Handle end of sequence entity
    if entity:
        entities.append(entity)
        entity_types.append(labels[entity[0]])

    return entities, entity_types
