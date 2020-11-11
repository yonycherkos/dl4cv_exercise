import numpy as np

def rank5_accuracy(preds, labels):
    rank1 = 0
    rank5 = 0
    for (pred, gt) in zip(preds, labels):
        pred = np.argsort(pred)[::-1]

        if gt in pred[:5]:
            rank5 += 1
        
        if gt == pred[0]:
            rank1 += 1

    rank1 /= float(len(labels))
    rank5 /= float(len(labels))

    return (rank1, rank5)