#!/usr/bin/env python3

# Adjusted from http://github.com/cisnlp/simalign/blob/master/scripts/calc_align_score.py

import argparse
from nltk.corpus import stopwords

STOPS = set(stopwords.words("english"))


def remove_stopwords(pairs, line, stopwords):
    pair_filtered = set()
    for pair in pairs:
        try:
            eng_token = line.split()[int(pair.split("-")[0])]
        except:
            continue
        if eng_token not in stopwords:
            pair_filtered.add(pair)
    return pair_filtered


def load_gold(g_path, text_path=None, wo_stopwords=True):
    gold_f = open(g_path, "r")
    pros = {}
    surs = {}
    all_count = 0.0
    surs_count = 0.0

    if text_path and wo_stopwords:
        text_file = open(text_path, "r")
        texts = text_file.readlines()

    for i, line in enumerate(gold_f):
        line = line.strip().split("\t")

        line[1] = line[1].split()

        pros[line[0]] = set([x.replace("p", "-") for x in line[1]])
        surs[line[0]] = set([x for x in line[1] if "p" not in x])
        if text_path and wo_stopwords:
            pros[line[0]] = remove_stopwords(
                pros[line[0]], texts[i].split("\t")[1], STOPS
            )
            surs[line[0]] = remove_stopwords(
                surs[line[0]], texts[i].split("\t")[1], STOPS
            )

        all_count += len(pros[line[0]])
        surs_count += len(surs[line[0]])

    return pros, surs, surs_count


def calc_score(input_path, probs, surs, surs_count, text_path=None, wo_stopwords=True):
    total_hit = 0.0
    p_hit = 0.0
    s_hit = 0.0
    target_f = open(input_path, "r")

    if text_path and wo_stopwords:
        text_file = open(text_path, "r")
        texts = text_file.readlines()

    for i, line in enumerate(target_f):
        line = line.strip().split("\t")

        if line[0] not in probs:
            continue
        if len(line) < 2:
            continue
        line[1] = line[1].split()
        if len(line[1][0].split("-")) > 2:
            line[1] = ["-".join(x.split("-")[:2]) for x in line[1]]

        if wo_stopwords:
            line[1] = remove_stopwords(line[1], texts[i].split("\t")[1], STOPS)

        p_hit += len(set(line[1]) & set(probs[line[0]]))
        s_hit += len(set(line[1]) & set(surs[line[0]]))
        total_hit += len(set(line[1]))
    target_f.close()

    y_prec = round(p_hit / max(total_hit, 1.0), 3)
    y_rec = round(s_hit / max(surs_count, 1.0), 3)
    y_f1 = round(2.0 * y_prec * y_rec / max((y_prec + y_rec), 0.01), 3)
    aer = round(1 - (s_hit + p_hit) / (total_hit + surs_count), 3)

    return y_prec, y_rec, y_f1, aer


if __name__ == "__main__":
    """
    Calculate alignment quality scores based on the gold standard.
    The output contains Precision, Recall, F1, and AER.
    The gold annotated file should be selected by "gold_path".
    The generated alignment file should be selected by "input_path".
    Both gold file and input file are in the FastAlign format with sentence number at the start of line separated with TAB.

    usage: python calc_align_score.py gold_file generated_file
    """

    parser = argparse.ArgumentParser(
        description="Calculate alignment quality scores based on the gold standard.",
        epilog="example: python calc_align_score.py gold_path input_path",
    )

    parser.add_argument("gold_path")
    parser.add_argument("input_path")
    parser.add_argument("text_path")
    args = parser.parse_args()

    probs, surs, surs_count = load_gold(args.gold_path, args.text_path, True)

    y_prec, y_rec, y_f1, aer = calc_score(
        f"{args.input_path}", probs, surs, surs_count, args.text_path, True
    )
