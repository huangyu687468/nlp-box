# coding=utf-8
import argparse
import codecs
import pdb


def main(args):
    print(args)

    labels = []
    with codecs.open(args.ground_truth, 'r', "utf8") as fin:
        for line in fin:
            label = int(line[0])
            labels.append(label)

    preds = []
    with codecs.open(args.prediction, 'r', "utf8") as fin:
        for line in fin:
            pred = int(line[0])
            preds.append(pred)

    tp, fp, tn, fn = 0, 0, 0, 0
    for label, pred in zip(labels, preds):
        if pred != 0 and label != 0:
            tp += 1
        if pred != 0 and label == 0:
            fp += 1
        if pred == 0 and label == 0:
            tn += 1
        if pred == 0 and label != 0:
            fn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)

    res_str = 'TP: %d  TN: %d  FP: %d  FN: %d\n' \
              'accuracy: %.6f\n' \
              'precision: %.6f\n' \
              'recall: %.6f\n' \
              'f1: %.6f\n\n\n' \
              % (tp, tn, fp, fn, accuracy, precision, recall, f1)
    print(res_str)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth", type=str, default='tmp/data/test/waimai_10k.csv')
    parser.add_argument("--prediction", type=str, default='tmp/result/test/test.txt')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
