import matplotlib.pyplot as plt
import numpy as np
import util
from pathlib import Path

DATA_DIR = Path(__file__).parent
PLOTS_DIR = DATA_DIR / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)


def main(train_path, save_path, reg = False):
    """Problem: Logistic regression with gradient descent.

    Args:
        train_path: Path to CSV file containing dataset for training.
        save_path: Path to save outputs; visualizations, predictions, etc.
    """
    x_train, y_train = util.load_csv(train_path, add_intercept=True)

    # *** START CODE HERE ***
    clf = LogisticRegression()
    clf.fit(x_train, y_train, reg)
    util.plot(x_train, y_train, clf.theta, save_path)

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
        self.reg_rate = reg_rate
    # *** END CODE HERE ***

    def boundary(self, x):
        """
        Outputs the y-values of the decision boundary (p(y|x) = 0.5) for the input x-values
        """    
        # *** START CODE HERE ***
        # 1/(1 + e^-z) = 0.5 iff z = 0
        # So, theta[0] + theta[1]x1 + theta[2]x2 = 0
        return -(self.theta[0] + self.theta[1]*x)/self.theta[2]
                
    # *** END CODE HERE ***

    def fit(self, x, y, reg = False):
        """Run gradient descent to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        if self.theta is None: self.theta = np.zeros(x.shape[1])

        for i in range(self.max_iter): 
            old_theta = self.theta
            self.theta = self.theta - self.learning_rate*self.loss_gradient(x, y, reg)
            if np.linalg.norm(old_theta - self.theta) < self.eps: break
            if i % 5000 == 0 and self.verbose: 
                print(f'Loss in iteration {i}: {self.loss(x, y, reg)}')
                print(f'Parameters: {self.theta}')
                print('==============================')

        if self.verbose:
            print(f'Loss in iteration {i}: {self.loss(x, y, reg)}')
            print(f'Parameters: {self.theta}')
            print('==============================') 
    # *** END CODE HERE ***
    
    def loss(self, x,y, reg):
        # *** START CODE HERE ***
        preds = self.predict(x)

        loss_no_reg = -sum(y*np.log(preds + self.eps) + (1-y)*np.log(1 - preds + self.eps))/x.shape[0]
        if not reg: return loss_no_reg       
        else: return loss_no_reg + (self.reg_rate/2)*(np.linalg.norm(self.theta)**2)

    # *** END CODE HERE ***
    
    def loss_gradient(self, x,y, reg):
        # *** START CODE HERE ***
        # Derived by chain rule d Ji/d theta_j = (d Ji/d h_theta)*(d h_theta/d z)*(d z/d theta_j)
        # where z = theta.T @ x. For regularized case it is trivial
        if not reg: return x.T @ (self.predict(x) - y)/x.shape[0]
        else: return x.T @ (self.predict(x) - y)/x.shape[0] + self.reg_rate*self.theta

    # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return 1/(1 + np.exp(-x @ self.theta))
    
    # *** END CODE HERE ***

if __name__ == '__main__':
    print('==== Training model on data set A ====')
    main(train_path=DATA_DIR / 'ds1_a.csv',
            save_path=PLOTS_DIR / 'logreg_pred_a_no-reg.png')

    print('\n==== Training model on data set B ====')
    main(train_path=DATA_DIR / 'ds1_b.csv',
            save_path=PLOTS_DIR / 'logreg_pred_b_no-reg.png')

    # *** START CODE HERE ***
    print('==== Training model on data set A (with regularisation)====')
    main(train_path=DATA_DIR / 'ds1_a.csv',
                save_path=PLOTS_DIR / 'logreg_pred_a_with-reg.png', reg = True)
    
    print('\n==== Training model on data set B (with regularisation)====')
    main(train_path=DATA_DIR / 'ds1_b.csv',
            save_path=PLOTS_DIR / 'logreg_pred_b_with-reg.png', reg = True)

# *** END CODE HERE ***