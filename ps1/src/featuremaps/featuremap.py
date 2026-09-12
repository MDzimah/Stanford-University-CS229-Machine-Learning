import matplotlib.pyplot as plt
import numpy as np
import util
import random
from pathlib import Path

np.seterr(all='raise')


factor = 2.0
DATA_DIR = Path(__file__).parent
PLOTS_DIR = DATA_DIR / 'plots'
EXPERIMENT_NUMBER = 0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None, k=None, sine=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta
        self.k = k
        self.sine = sine

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # To fit just means to find the best theta for the model given the training
        # inputs and outputs
        X_hat = self.create_sin(self.k, X) if self.sine else self.create_poly(self.k, X)
        self.theta = np.linalg.solve(X_hat.T @ X_hat, X_hat.T @ y)

# *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        return X[:, 1, None]**np.arange(0, k + 1)

# *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        return np.concatenate((self.create_poly(k, X), np.sin(X[:,1, None])), axis = 1)

# *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        if self.sine == False: return self.create_poly(self.k, X) @ self.theta
        else: return self.create_sin(self.k, X) @ self.theta
# * END CODE HERE ***


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png', print_theta=False):
    global EXPERIMENT_NUMBER
    EXPERIMENT_NUMBER += 1
    PLOTS_DIR.mkdir(exist_ok=True)
    filename = PLOTS_DIR / f'ex_{EXPERIMENT_NUMBER}_{Path(filename).name}'

    train_x,train_y=util.load_dataset(train_path,add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)
    colors = list(plt.cm.tab20.colors)
    random.Random(EXPERIMENT_NUMBER).shuffle(colors)

    for index, k in enumerate(ks):
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        lm = LinearModel(k = k, sine = sine)
        lm.fit(train_x, train_y)
        plot_y = lm.predict(plot_x)

# *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k, color=colors[index % len(colors)])

    plt.legend()
    plt.savefig(filename)
    plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all expetriments
    '''
    run_exp(train_path, ks = [3], filename= DATA_DIR / 'plotDeg3Poly.png')
    run_exp(train_path, ks = [3, 5, 10, 20], filename = DATA_DIR / 'plotDegkPoly.png')
    run_exp(train_path, True, [0, 1, 2, 3, 5, 10, 20], DATA_DIR / 'plotDegkPoly&SineWave.png')
    run_exp(small_path, False, [1, 2, 5, 10, 20], DATA_DIR / 'plotDegkPolySmallData.png')

    
# *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path=DATA_DIR / 'train.csv',
        small_path=DATA_DIR / 'small.csv',
        eval_path=DATA_DIR / 'test.csv')
