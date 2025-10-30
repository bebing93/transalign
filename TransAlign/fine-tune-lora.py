import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from collections import defaultdict
import random

from models.nllb_encoder_model import M2M100EncoderModel
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType


class AlignmentDataset(Dataset):
    def __init__(self, dataset_file):
        self.data = []
        self.tgt_lang_indices = defaultdict(list)
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                example = json.loads(line)
                self.data.append(example)
                tgt_lang = example['tgt_lang']
                self.tgt_lang_indices[tgt_lang].append(idx)
        self.tgt_langs = list(self.tgt_lang_indices.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class LanguageBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        # Create a copy of indices for each language
        self.tgt_lang_indices = {lang: indices.copy() for lang, indices in dataset.tgt_lang_indices.items()}
        self.languages = list(self.tgt_lang_indices.keys())

    def __iter__(self):
        languages = self.languages.copy()
        lang_indices = {lang: indices.copy() for lang, indices in self.tgt_lang_indices.items()}

        while languages:
            # Randomly select a language
            lang = random.choice(languages)
            indices = lang_indices[lang]

            if len(indices) >= self.batch_size:
                batch_indices = random.sample(indices, self.batch_size)
            else:
                # Sample with replacement if not enough samples
                batch_indices = random.choices(indices, k=self.batch_size)
                languages.remove(lang)

            # Remove used indices to prevent duplicates
            for idx in batch_indices:
                if idx in indices:
                    indices.remove(idx)
            if not indices and lang in languages:
                languages.remove(lang)

            yield batch_indices

    def __len__(self):
        # Approximate total number of batches
        total_samples = sum(len(indices) for indices in self.dataset.tgt_lang_indices.values())
        return (total_samples + self.batch_size - 1) // self.batch_size


def nllb_collate_fn(batch):
    # All items in batch have 'tgt_lang'; 'src_lang' is always English
    src_lang = 'eng_Latn'  # Since source language is always English
    tgt_lang = batch[0]['tgt_lang']

    # Instantiate tokenizers with appropriate language codes
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer_src = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
    tokenizer_tgt = AutoTokenizer.from_pretrained(model_name, src_lang=tgt_lang)  # Using src_lang because we're using the encoder "tgt" language refers to the decoder part

    # Extract source and target texts, as well as the respective (precalculated NLLB) gold-matrices
    source_texts = [item['source_text'] for item in batch]
    target_texts = [item['target_text'] for item in batch]
    gold_matrices = [item['gold_matrices'] for item in batch]

    # Tokenize source and target sentences
    source_inputs = tokenizer_src(source_texts, padding=True, truncation=True, return_tensors='pt')
    target_inputs = tokenizer_tgt(target_texts, padding=True, truncation=True, return_tensors='pt')

    # nested list for every sentence with tokenized src and trg sentences
    words_tokens = []

    for idx, item in enumerate(batch):
        l1_tokens = [tokenizer_src.tokenize(word) for word in source_texts[idx].split()]
        l2_tokens = [tokenizer_tgt.tokenize(word) for word in target_texts[idx].split()]
        words_tokens.append([l1_tokens, l2_tokens])

    # flattened words into subwords with map to revert subwords back together
    sentences_bpe_lists = []
    sentences_b2w_map = []

    for idx in range(len(words_tokens)):
        sent_pair = [[bpe for w in sent for bpe in w] for sent in words_tokens[idx]]
        b2w_map_pair = [[i for i, w in enumerate(sent) for bpe in w] for sent in words_tokens[idx]]
        sentences_bpe_lists.append(sent_pair)
        sentences_b2w_map.append(b2w_map_pair)

    # Split obtained lists into source and target specific lists
    source_sentences_bpe_lists = []
    target_sentences_bpe_lists = []
    source_sentences_b2w_map = []
    target_sentences_b2w_map = []

    for i in range(len(sentences_bpe_lists)):
        source_sentences_bpe_lists.append(sentences_bpe_lists[i][0])
        target_sentences_bpe_lists.append(sentences_bpe_lists[i][1])
        source_sentences_b2w_map.append(sentences_b2w_map[i][0])
        target_sentences_b2w_map.append(sentences_b2w_map[i][1])

    # Load gold matrices as tensors
    tensor_gold_matrices = []
    for i in range(len(batch)):
        gm_tensor = torch.tensor(gold_matrices[i], dtype=torch.float)
        tensor_gold_matrices.append(gm_tensor)

    batch = {
        'source_inputs': source_inputs,
        'source_sentences_bpe_lists': source_sentences_bpe_lists,
        'source_sentences_b2w_map': source_sentences_b2w_map,
        'target_inputs': target_inputs,
        'target_sentences_bpe_lists': target_sentences_bpe_lists,
        'target_sentences_b2w_map': target_sentences_b2w_map,
        'gold_matrix': tensor_gold_matrices,
        'trg_lang': tgt_lang
    }

    return batch

def labse_collate_fn(batch):
    # All items in batch have 'tgt_lang'; 'src_lang' is always English
    src_lang = 'eng_Latn'  # Since source language is always English
    tgt_lang = batch[0]['tgt_lang']

    # Instantiate tokenizers with appropriate language codes
    model_name = 'sentence-transformers/LaBSE'
    tokenizer_src = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
    tokenizer_tgt = AutoTokenizer.from_pretrained(model_name, src_lang=tgt_lang)  # Using src_lang because we're using the encoder "tgt" language refers to the decoder part

    # Extract source and target texts, as well as the respective (precalculated NLLB) gold-matrices
    source_texts = [item['source_text'] for item in batch]
    target_texts = [item['target_text'] for item in batch]
    gold_matrices = [item['gold_matrices'] for item in batch]

    # Tokenize source and target sentences
    source_inputs = tokenizer_src(source_texts, padding=True, truncation=True, return_tensors='pt')
    target_inputs = tokenizer_tgt(target_texts, padding=True, truncation=True, return_tensors='pt')

    # nested list for every sentence with tokenized src and trg sentences
    words_tokens = []

    for idx, item in enumerate(batch):
        l1_tokens = [tokenizer_src.tokenize(word) for word in source_texts[idx].split()]
        l2_tokens = [tokenizer_tgt.tokenize(word) for word in target_texts[idx].split()]
        words_tokens.append([l1_tokens, l2_tokens])

    # flattened words into subwords with map to revert subwords back together
    sentences_bpe_lists = []
    sentences_b2w_map = []

    for idx in range(len(words_tokens)):
        sent_pair = [[bpe for w in sent for bpe in w] for sent in words_tokens[idx]]
        b2w_map_pair = [[i for i, w in enumerate(sent) for bpe in w] for sent in words_tokens[idx]]
        sentences_bpe_lists.append(sent_pair)
        sentences_b2w_map.append(b2w_map_pair)

    # Split obtained lists into source and target specific lists
    source_sentences_bpe_lists = []
    target_sentences_bpe_lists = []
    source_sentences_b2w_map = []
    target_sentences_b2w_map = []

    for i in range(len(sentences_bpe_lists)):
        source_sentences_bpe_lists.append(sentences_bpe_lists[i][0])
        target_sentences_bpe_lists.append(sentences_bpe_lists[i][1])
        source_sentences_b2w_map.append(sentences_b2w_map[i][0])
        target_sentences_b2w_map.append(sentences_b2w_map[i][1])

    # Load gold matrices as tensors
    tensor_gold_matrices = []
    for i in range(len(batch)):
        gm_tensor = torch.tensor(gold_matrices[i], dtype=torch.float)
        tensor_gold_matrices.append(gm_tensor)

    batch = {
        'source_inputs': source_inputs,
        'source_sentences_bpe_lists': source_sentences_bpe_lists,
        'source_sentences_b2w_map': source_sentences_b2w_map,
        'target_inputs': target_inputs,
        'target_sentences_bpe_lists': target_sentences_bpe_lists,
        'target_sentences_b2w_map': target_sentences_b2w_map,
        'gold_matrix': tensor_gold_matrices,
        'trg_lang': tgt_lang
    }

    return batch



def set_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# -------------------------------
# Training Loop
# -------------------------------
def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed = args.seed
    set_seeds(seed)

    # Load dataset
    dataset = AlignmentDataset(args.dataset_file)

    # Initialize the custom sampler
    sampler = LanguageBatchSampler(dataset, batch_size=args.batch_size)

    # Create the DataLoader
    model_path = args.model
    if "nllb" in model_path:
        dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=nllb_collate_fn)
    elif "LaBSE" in model_path:
        dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=labse_collate_fn)
    else:
        raise NotImplementedError

    # Initialize model and move to device, set to train mode
    model_path = args.model

    if "nllb" in model_path:
        model = M2M100EncoderModel.from_pretrained(model_path, output_hidden_states=True)
        model.to(device)
        model.train()
        model_name = model_path.split("/")[-1] 
    elif "LaBSE" in model_path:
        model = AutoModel.from_pretrained(model_path, output_hidden_states=True)
        print(model)
        model.to(device)
        model.train()
        model_name = model_path.split("/")[-1]
    else:
        raise NotImplementedError
    
    if "nllb" in model_path:
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["fc1", "fc2"])
    elif "LaBSE" in model_path:
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["intermediate.dense", "output.dense"])
    else:
        raise NotImplementedError


    model = get_peft_model(model, lora_config)
    print(model)
    model.print_trainable_parameters() 
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    # Start of training epoch loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        # Shuffle target languages for each epoch
        # For each epoch we go through all languages in a random order.
        random.shuffle(dataset.tgt_langs)

        # Take all examples from one of the target languages until all completed

        for batch in tqdm(dataloader):
            src_lang = 'eng_Latn'  # Source language is always English
            tgt_lang = batch['trg_lang']
            print(f"Training on target language: {tgt_lang}")

            optimizer.zero_grad()

            # Move tensors to device
            batch_source = batch['source_inputs'].to(device)
            batch_target = batch['target_inputs'].to(device)
            batch_gold_matrices = batch['gold_matrix']
            batch_source_sentences_bpe_lists = batch['source_sentences_bpe_lists']
            batch_target_sentences_bpe_lists = batch['target_sentences_bpe_lists']
            # Forward passes through the encoder
            source_outputs = model(**batch_source)["hidden_states"]
            target_outputs = model(**batch_target)["hidden_states"]

            # extract hidden specific layer from hidden states and remove first and last token
            hidden_states_src_layer_batch = source_outputs[args.layer][:, 1:-1, :]
            hidden_states_trg_layer_batch = target_outputs[args.layer][:, 1:-1, :]

            # source_hidden_states = source_outputs.last_hidden_state  # [batch_size, src_seq_len, hidden_size]
            # target_hidden_states = target_outputs.last_hidden_state  # [batch_size, tgt_seq_len, hidden_size]

            # normalized_hidden_states_src_layer_batch = F.normalize(hidden_states_src_layer_batch, dim=2)
            # normalized_hidden_states_trg_layer_batch = F.normalize(hidden_states_trg_layer_batch, dim=2)

            # batch_sim = torch.bmm(normalized_hidden_states_src_layer_batch, torch.transpose(normalized_hidden_states_trg_layer_batch, 1, 2))
            # batch_sim = ((batch_sim + 1.0) / 2.0)  # .cpu().detach().numpy()

            # vectors = [normalized_hidden_states_src_layer_batch[in_batch_id, :len(sent_pair[0])], normalized_hidden_states_trg_layer_batch[in_batch_id, :len(sent_pair[1])]]

            # sim = batch_sim[in_batch_id, :len(sent_pair[0]), :len(sent_pair[1])]

            # Initialize batch loss
            batch_loss = 0

            for i in range(len(batch_gold_matrices)):

                source_hidden_states = hidden_states_src_layer_batch[i, :len(batch_source_sentences_bpe_lists[i])]  # .numpy()
                target_hidden_states = hidden_states_trg_layer_batch[i, :len(batch_target_sentences_bpe_lists[i])]  # .numpy()

                s_xy = torch.softmax(torch.matmul(source_hidden_states, target_hidden_states.T), dim=1)
                s_yx = torch.softmax(torch.matmul(source_hidden_states, target_hidden_states.T), dim=0)
                gold_matrix = batch_gold_matrices[i]  # [src_len, tgt_len] for the nllb tokenizer

                # Sum of all entries in the forward and backward similarity matrices
                flat_s_xy = s_xy.view(-1)
                flat_s_yx = s_yx.view(-1)
                flat_gold_matrix = gold_matrix.view(-1)

                flat_gold_matrix.sum() * 0.5
                for idx, value in enumerate(flat_gold_matrix):
                    batch_loss -= (flat_gold_matrix[idx] * 0.5 * (flat_s_xy[idx]/len(batch_source_sentences_bpe_lists[i]) + flat_s_yx/len(batch_target_sentences_bpe_lists[i]))).sum()

            # Backward pass and optimization
            batch_loss.backward()
            optimizer.step()

            print(f"Batch Loss: {batch_loss.item():.4f}")

        # Save model checkpoint after each epoch
        checkpoint_dir = Path(args.output_dir) / f"lora-r8-a32-d0.1-fc1fc2_epoch_{epoch+1}_{seed}_{model_name}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        print(f"Model checkpoint saved to {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="NLLB Alignment Training on Gold Word Alignments.")
    parser.add_argument('dataset_file', type=str, help='Path to the dataset JSON file.')
    parser.add_argument('model', type=str, help='Model for fine-tuning')

    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
    parser.add_argument('--layer', type=int, default=12, help='What layer to extract alignments for training.')
    parser.add_argument('--output_dir', type=str, default='lora-r8-a32-d0.1-fc1fc2', help='Directory to save model checkpoints.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducability')

    args = parser.parse_args()
    print(args.layer)
    train_model(args)


if __name__ == '__main__':
    main()
