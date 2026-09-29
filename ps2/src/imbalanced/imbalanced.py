import numpy as np
import util
import sys
from random import random
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent
PLOTS_DIR = DATA_DIR / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)

sys.path.append(str(DATA_DIR.parent / 'logreg_stability'))

### NOTE : You need to complete logreg implementation first! If so, make sure to set the regularization weight to 0.
from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Ratio of class 0 to class 1
kappa = 0.1

def main(train_path, validation_path, save_path):
    """Problem 2: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    save_path = Path(save_path)
    output_path_naive = PLOTS_DIR / save_path.name.replace(WILDCARD, 'naive').replace('.txt', '.png')
    output_path_upsampling = PLOTS_DIR / save_path.name.replace(WILDCARD, 'upsampling').replace('.txt', '.png')

    # *** START CODE HERE ***
    x_train, y_train = util.load_dataset(train_path)
    x_val, y_val = util.load_dataset(validation_path)

    clf = LogisticRegression(verbose = False)
    clf.fit(x_train, y_train, reg = False)
    preds = clf.predict(x_val)

    TP = TN = FP = FN = 0
    for i in range(y_val.shape[0]):
        if y_val[i] == 1: 
            if preds[i] >= 0.5: TP += 1
            else: FN += 1   
        else:
            if preds[i] < 0.5: TN += 1
            else: FP += 1

    ex = TP + TN + FP + FN
    A1 = TP/(TP + FN)
    A0 = TN/(TN + FP)
    ro = (TP + FN)/ex
    print(f'Classifier\'s accuracy (A): {ro*A1 + (1-ro)*A0}')
    print(f'Balanced accuracy: {(A1 + A0)/2}')
    print(f'A1: {A1}')
    print(f'A0: {A0}')
    util.plot(x_val, y_val, clf.theta, output_path_naive.with_suffix(".png"))

    repeat_counts = np.where(y_train == 1, int(1 / kappa), 1)
    x_train = np.repeat(x_train, repeat_counts, axis = 0) # We have 2 dimensional features, hence axis = 0
    y_train = np.repeat(y_train, repeat_counts)

    clf_reweight = LogisticRegression(verbose = False)
    clf_reweight.fit(x_train, y_train, reg = False)
    preds = clf_reweight.predict(x_val)
    
    TP = TN = FP = FN = 0
    for i in range(y_val.shape[0]):
        if y_val[i] == 1: 
            if preds[i] >= 0.5: TP += 1
            else: FN += 1   
        else:
            if preds[i] < 0.5: TN += 1
            else: FP += 1

    A1 = TP/(TP + FN)
    A0 = TN/(TN + FP)
    ro = (TP + FN)/ex
    print('==========\nRE-WEIGHTING MINORITY CLASS')
    print(f'Classifier\'s accuracy (A): {ro*A1 + (1-ro)*A0}')
    print(f'Balanced accuracy: {(A1 + A0)/2}')
    print(f'A1: {A1}')
    print(f'A0: {A0}')
    util.plot(x_val, y_val, clf_reweight.theta, output_path_upsampling.with_suffix(".png"))


# *** END CODE HERE

if __name__ == '__main__':
    main(train_path=DATA_DIR / 'train.csv',
        validation_path=DATA_DIR / 'validation.csv',
        save_path=DATA_DIR / 'imbalanced_X_pred.txt')
