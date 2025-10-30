import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path to import common_utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_data

import evaluate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_bio(
    pred_file_path: str,
    true_file_path: str,
    out_score_path: str = None,
) -> float:
    """
    Evaluate BIO tagging performance.

    Args:
        pred_file_path: Path to file with predicted labels (test.txt format)
        true_file_path: Path to file with true labels (test.txt format)
        out_score_path: Optional path to save evaluation score

    Returns:
        F1 score as a float
    """
    logger.info(f"Loading predictions from {pred_file_path}")
    pred_labels = load_data(pred_file_path, with_labels=True)

    logger.info(f"Loading true labels from {true_file_path}")
    true_labels = load_data(true_file_path, with_labels=True)

    if len(pred_labels) != len(true_labels):
        raise ValueError(
            f"Prediction and label length mismatch: {len(pred_labels)} sentences vs {len(true_labels)} sentences"
        )

    # Verify sentence-level alignment
    for i, (pred, true) in enumerate(zip(pred_labels, true_labels)):
        if len(pred) != len(true):
            raise ValueError(
                f"Sentence {i}: Token count mismatch - {len(pred)} predicted vs {len(true)} true labels"
            )

    metric = evaluate.load("seqeval", keep_in_memory=True)
    score = metric.compute(predictions=pred_labels, references=true_labels)
    f1_score = round(score["overall_f1"] * 100, 2)

    logger.info(f"F1 Score: {f1_score}%")
    logger.info(f"Overall Precision: {round(score['overall_precision'] * 100, 2)}%")
    logger.info(f"Overall Recall: {round(score['overall_recall'] * 100, 2)}%")
    logger.info(f"Overall Accuracy: {round(score['overall_accuracy'] * 100, 2)}%")

    if out_score_path:
        with open(out_score_path, "w", encoding="utf-8") as f:
            f.write(f"{f1_score}\n")
        logger.info(f"Evaluation results saved to {out_score_path}")

    return f1_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BIO tagging performance")
    parser.add_argument(
        "pred_file", help="Path to file with predicted labels (test.txt format)"
    )
    parser.add_argument(
        "true_file", help="Path to file with true labels (test.txt format)"
    )
    parser.add_argument(
        "--out-score-path", default=None, help="Path to save evaluation score"
    )

    args = parser.parse_args()

    try:
        evaluate_bio(
            pred_file_path=args.pred_file,
            true_file_path=args.true_file,
            out_score_path=args.out_score_path,
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
