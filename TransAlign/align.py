# Adjusted from https://github.com/cisnlp/simalign/blob/master/simalign/simalign.py

import argparse
import os
import torch
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)
from peft import PeftModel


class NeurAligner(object):

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        layer: int = 0,
        device=torch.device("cpu"),
        lang1: str = None,
        lang2: str = None,
        lora_path: str = None,
        threshold: float = None,
    ):

        device = torch.device("cuda") if torch.cuda.is_available() else device

        self.model = self.load_model(model_name, device)

        if lora_path is not None:
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model.eval()
            self.model.to(device)
            print(self.model)

        # Load tokenizers
        self.tokenizer_lang1 = AutoTokenizer.from_pretrained(
            model_name, src_lang=lang1, add_prefix_space=True
        )
        self.tokenizer_lang2 = AutoTokenizer.from_pretrained(
            model_name, src_lang=lang2, add_prefix_space=True
        )
        self.device = device
        self.layer = layer
        self.lang1 = lang1
        self.lang2 = lang2
        self.threshold = threshold

    @staticmethod
    def load_model(model_name: str, device):
        config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        full_model = AutoModel.from_pretrained(model_name, config=config)

        # For encoder-decoder models, extract just the encoder
        if hasattr(full_model, "get_encoder"):
            emb_model = full_model.get_encoder()
        else:
            # For encoder-only models, use the model as-is
            emb_model = full_model

        emb_model.eval()
        emb_model.to(device)
        return emb_model

    def get_hidden_states(
        self, sent_batch: list[list[str]], is_lang1: bool = False
    ) -> tuple[torch.Tensor]:
        with torch.no_grad():
            tokenizer = self.tokenizer_lang1 if is_lang1 else self.tokenizer_lang2
            inputs = tokenizer(
                sent_batch,
                is_split_into_words=True,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            return self.model(**inputs.to(self.device))["hidden_states"]

    def get_softmax(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        forward = torch.softmax(torch.matmul(x, y.T), dim=1)
        backward = torch.softmax(torch.matmul(x, y.T), dim=0)
        final = (forward > self.threshold) * (backward > self.threshold)
        return final


def load_corpus(path: str) -> list[list[str]]:
    """Load a corpus file in test.txt format, returning list of word lists.

    Format: Each line is 'token' or 'token label', blank lines separate sentences.
    Labels are ignored if present.
    """
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        tokens = []
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(tokens)
                    tokens = []
            else:
                parts = line.split()
                if parts:
                    # Take only the first part (token), ignore label if present
                    tokens.append(parts[0])
        if tokens:
            sentences.append(tokens)
    return sentences


# Press the green button in the gutter to run the script.
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Extract word alignments using transformer embeddings",
        epilog="Example: python3 main.py lang1.txt lang2.txt --model-name bert-base-multilingual-cased --layer 8 --softmax 0.001",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "lang1_path",
        type=str,
        help="Path to first language file (one sentence per line, optionally tab-separated with index)",
    )
    parser.add_argument(
        "lang2_path",
        type=str,
        help="Path to second language file (same format as first language file)",
    )

    # Model configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-multilingual-cased",
        help="Hugging Face model name or path (e.g., 'bert-base-multilingual-cased', 'xlm-roberta-base', 'facebook/nllb-200-distilled-600M')",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer number to extract embeddings from (required)",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA adapter checkpoint to load on top of the base model",
    )

    # Language-specific settings
    parser.add_argument(
        "--lang1",
        type=str,
        default="eng_Latn",
        help="First language code for NLLB tokenizer (e.g., 'eng_Latn', 'fra_Latn', 'deu_Latn'). Default: 'eng_Latn'",
    )
    parser.add_argument(
        "--lang2",
        type=str,
        default="eng_Latn",
        help="Second language code for NLLB tokenizer (e.g., 'eng_Latn', 'fra_Latn', 'deu_Latn'). Default: 'eng_Latn'",
    )

    # Extraction method
    parser.add_argument(
        "--softmax",
        type=float,
        metavar="THRESHOLD",
        help="Use softmax-based alignment extraction with specified threshold (e.g., 0.001)",
    )

    # Output configuration
    parser.add_argument(
        "--output-path",
        type=str,
        default="./alignments",
        help="Output directory path where alignment files will be saved",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=f"alignment.txt",
        help="Output directory path where alignment files will be saved",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of sentence pairs to process per batch",
    )

    args = parser.parse_args()

    neuralign = NeurAligner(
        model_name=args.model_name,
        layer=args.layer,
        lang1=args.lang1,
        lang2=args.lang2,
        lora_path=args.lora_path,
        threshold=args.softmax,
    )

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    # Load corpora (already split into words)
    corpus_lang1 = load_corpus(args.lang1_path)
    corpus_lang2 = load_corpus(args.lang2_path)

    # Batch tokenize to get word-to-subword mappings
    tokens_lang1_all = neuralign.tokenizer_lang1(
        corpus_lang1,
        is_split_into_words=True,
        add_special_tokens=False,
        padding=False,
    )
    tokens_lang2_all = neuralign.tokenizer_lang2(
        corpus_lang2,
        is_split_into_words=True,
        add_special_tokens=False,
        padding=False,
    )

    # Process tokenization results
    sentences_bpe_lengths = []
    sentences_b2w_map = []

    for sent_id in range(len(corpus_lang1)):
        # Get token lengths and word_ids mapping
        len_lang1 = len(tokens_lang1_all["input_ids"][sent_id])
        len_lang2 = len(tokens_lang2_all["input_ids"][sent_id])

        b2w_map_lang1 = tokens_lang1_all.word_ids(sent_id)
        b2w_map_lang2 = tokens_lang2_all.word_ids(sent_id)

        sentences_bpe_lengths.append([len_lang1, len_lang2])
        sentences_b2w_map.append([b2w_map_lang1, b2w_map_lang2])

    # Custom collate function to handle variable-length word lists
    def collate_fn(batch):
        indices, lang1_words, lang2_words = zip(*batch)
        return list(indices), list(lang1_words), list(lang2_words)

    ds = [
        (idx, corpus_lang1[idx], corpus_lang2[idx]) for idx in range(len(corpus_lang1))
    ]
    data_loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    with open(os.path.join(args.output_path, args.output_file), "w") as out_f:
        for batch_id, batch_sentences in enumerate(tqdm(data_loader)):

            hidden_states_lang1_batch = neuralign.get_hidden_states(
                batch_sentences[1], is_lang1=True
            )
            hidden_states_lang2_batch = neuralign.get_hidden_states(
                batch_sentences[2], is_lang1=False
            )

            # Extract hidden states for the specified layer and remove first and last token
            hidden_states_lang1_layer_batch = hidden_states_lang1_batch[
                neuralign.layer
            ][:, 1:-1, :]
            hidden_states_lang2_layer_batch = hidden_states_lang2_batch[
                neuralign.layer
            ][:, 1:-1, :]

            for in_batch_id, sent_id in enumerate(batch_sentences[0]):

                sent_lengths = sentences_bpe_lengths[sent_id]

                softsimx = hidden_states_lang1_layer_batch[
                    in_batch_id, : sent_lengths[0]
                ]
                softsimy = hidden_states_lang2_layer_batch[
                    in_batch_id, : sent_lengths[1]
                ]
                alignment_mat = neuralign.get_softmax(x=softsimx, y=softsimy)

                # Extract word-level alignments
                b2w_aligns = set()
                for i in range(alignment_mat.shape[0]):
                    for j in range(alignment_mat.shape[1]):
                        if alignment_mat[i, j] > 0:
                            b2w_aligns.add(
                                "{}-{}".format(
                                    sentences_b2w_map[sent_id][0][i],
                                    sentences_b2w_map[sent_id][1][j],
                                )
                            )

                out_f.write(str(sent_id) + "\t" + " ".join(sorted(b2w_aligns)) + "\n")
