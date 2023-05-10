#!/usr/bin/env python3

import sys
import csv


def fin(positives, negatives, max_neg, max_pos, writer):
    positives.sort(key=lambda t: -t[0])
    negatives.sort(key=lambda t: t[0])
    for _, row in positives[:max_pos]:
        writer.writerow(row)
    for _, row in negatives[:max_neg]:
        writer.writerow(row)


def main():
    skip_wo_pos_or_neg = 0
    max_unique_doc_id_1 = 10_000
    max_negatives = 5
    max_positives = 5
    meta_path = sys.argv[1]
    with (
        open(meta_path, 'r', encoding='utf8') as fp,
        open(meta_path + '.balanced', 'w', encoding='utf8') as outfp,
    ):
        reader = csv.reader(fp)
        writer = csv.writer(outfp)

        writer.writerow(next(reader))

        cur_src_id = None
        positive_examples = []
        negative_examples = []
        written = 0
        for row in reader:
            src_id, _, _, _, label, rating = row
            if cur_src_id is None:
                cur_src_id = src_id
            if cur_src_id != src_id:
                if not positive_examples or not negative_examples:
                    # Do not add examples without positive or negative
                    skip_wo_pos_or_neg += 1
                    cur_src_id = src_id
                    positive_examples = []
                    negative_examples = []
                    continue
                fin(
                    positive_examples,
                    negative_examples,
                    max_neg=max_negatives,
                    max_pos=max_positives,
                    writer=writer,
                )
                cur_src_id = src_id
                positive_examples = []
                negative_examples = []
                written += 1
                if written >= max_unique_doc_id_1:
                    break

            if int(label) == 1:
                positive_examples.append((float(rating), row))
            elif int(label) == 0:
                negative_examples.append((float(rating), row))
            else:
                raise RuntimeError("Unknown label")
        if positive_examples and negative_examples:
            fin(
                positive_examples,
                negative_examples,
                max_neg=max_negatives,
                max_pos=max_positives,
                writer=writer,
            )
        print("skip_wo_pos_or_neg", skip_wo_pos_or_neg)


if __name__ == '__main__':
    main()
