import numpy as np
import util
import sys
from random import random
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent

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
    output_path_naive = save_path.with_name(save_path.name.replace(WILDCARD, 'naive'))
    output_path_upsampling = save_path.with_name(save_path.name.replace(WILDCARD, 'upsampling'))

    # *** START CODE HERE ***
    
# TODO: implement this section.
pass
# *** END CODE HERE

if __name__ == '__main__':
    main(train_path=DATA_DIR / 'train.csv',
        validation_path=DATA_DIR / 'validation.csv',
        save_path=DATA_DIR / 'imbalanced_X_pred.txt')
