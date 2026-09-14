import numpy as np
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import util
from pathlib import Path

DATA_DIR = Path(__file__).parent

def generate_plot(betas, X, Y, X_val, Y_val, save_path):
    """Generate a scatter plot of validation error vs. norm

    Args:
        betas: list of numpy arrays, indicating different solutions
        X, Y: training dataset 
        X_val, Y_val: validataion dataset
        save_path: path to save the plot
    """
    # check if the validationerror is zero
    for b in betas:
        assert(np.allclose(X.dot(b), Y))

    # compute the norm and the validation error of all the solutions in list beta
    val_err = []
    norms = []
    for i in range(len(betas)):
        val_err.append(np.mean((X_val.dot(betas[i]) - Y_val) ** 2))
        norms.append(np.linalg.norm(betas[i]))

    print(norms)
    # plot the validation error against norm of the solution
    util.plot_points(norms, val_err, save_path)

def linear_model_main():
    plots_dir = DATA_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)
    save_path_linear = plots_dir / "impreg_lin.png"
    
    train_path = DATA_DIR / 'ir1_train.csv'
    valid_path = DATA_DIR / 'ir1_valid.csv'
    X, Y = util.load_dataset(train_path)
    X_val, Y_val = util.load_dataset(valid_path)

    # *** START CODE HERE ***

    # beta_0 = X.T @ (X @ X.T)^-1 @ y iff X @ X.T @ (X^-1)^T @ beta_0 
    # = X @ X.T @ v = beta_0
    v = np.linalg.solve(X @ X.T, Y)
    beta_0 = X.T @ v

    assert(np.allclose(X.dot(beta_0), Y))

    ns = null_space(X)
    betas = [beta_0.copy()]
    for j in [0, 1, 2]:
        betas.append(beta_0 + ns[:, j])
    generate_plot(betas, X, Y, X_val, Y_val, save_path_linear)
    
# *** END CODE HERE ***

if __name__ == '__main__':
    linear_model_main()
