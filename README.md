# TransAlign: Machine Translation Encoders are Strong Word Aligners, Too
Simple label projection for BIO-scheme data with machine translation encoders as presented in our paper "TransAlign: Machine Translation Encoders are Strong Word Aligners, Too"

### Dependencies

Start by installing the dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

### Simple Label Project for BIO-scheme data

1. Produce the pairwise word alignments
```bash
python TransAlign/align.py \
    sample_data/test-bam.txt \
    sample_data/test-translate-bam-en-preds.txt \
    --model-name "facebook/nllb-200-distilled-600M" \
    --layer 12 \
    --lang1 bam_Latn \
    --lang2 eng_Latn \
    --softmax 0.001 \
    --lora-path checkpoints/nllb-200-distilled-600M/ \
    --output-path "sample_data" \
    --output-file bam-en-alignments.txt
```

2. Project the labels
```bash
python TransAlign/evaluation/project_labels.py \
    ./sample_data/test-translate-bam-en-preds.txt \
    ./sample_data/test-bam.txt \
    ./sample_data/bam-en-alignments.txt \
    ./sample_data/test-translate-bam-en-projected.txt
```

3. Run evaluation
```bash
python TransAlign/evaluation/evaluate_bio.py \
    ./sample_data/test-translate-bam-en-projected.txt \
    ./sample_data/test-bam.txt \
    --out-score-path ./sample_data/bam-score.txt
```

### Downstream training and evaluation data

The downstream training and evaluation data, can be found [here](https://github.com/bebing93/devil-in-details)

### TODO

- Add training data and fine-tuning script for word aligners