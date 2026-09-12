import matplotlib.pyplot as plt
import numpy as np
import util
from pathlib import Path

DATA_DIR = Path(__file__).parent


def main(train_path, save_path):
    """Problem: Logistic regression with gradient descent.

    Args:
        train_path: Path to CSV file containing dataset for training.
        save_path: Path to save outputs; visualizations, predictions, etc.
    """
    x_train, y_train = util.load_csv(train_path, add_intercept=True)

    # *** START CODE HERE ***
    
# TODO: implement this section.
pass
# *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression using gradient descent.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, learning_rate=1, max_iter=100000, eps=1e-5,
                 theta_0=None, verbose=True, reg_rate=0.01):
        """
        Args:
            learning_rate: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

        # *** START CODE HERE ***
        
# TODO: implement this section.
pass
# *** END CODE HERE ***
        
    
    def boundary(self, x):
        """
        Outputs the y-values of the decision boundary (p(y|x) = 0.5) for the input x-values
        """    
        if self.theta is None:
            self.theta = np.zeros(x.shape[1])

        return (-self.theta[0] - self.theta[1] * x) / self.theta[2]

    def fit(self, x, y):
        """Run gradient descent to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        
# TODO: implement this section.
pass
# *** END CODE HERE ***
    
    def loss(self, x,y):
        if self.theta is None:
            return np.zeros(x.shape[1])
        
        preds = self.predict(x)
        return - np.sum(y * np.log(preds + 1e-5) + (1-y) * np.log(1 - preds+ 1e-5), axis=0) + 0.5 * self.reg_rate * np.linalg.norm(self.theta[1:]) ** 2
    
    def loss_gradient(self, x,y):
        if self.theta is None:
            return np.zeros(x.shape[1])
        preds = self.predict(x)
        grad = x.T @ (preds - y) + self.reg_rate * np.concatenate(([0], self.theta[1:]))
        return grad

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        
# TODO: implement this section.
pass
# *** END CODE HERE ***

if __name__ == '__main__':
    print('==== Training model on data set A ====')
        main(train_path=DATA_DIR / 'ds1_a.csv',
            save_path=DATA_DIR / 'logreg_pred_a_no-reg.png')

    print('\n==== Training model on data set B ====')
        main(train_path=DATA_DIR / 'ds1_b.csv',
            save_path=DATA_DIR / 'logreg_pred_b_no-reg.png')
