import numpy as np
import util
import matplotlib.pyplot as plt

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    
    # *** START CODE HERE ***
    p = PoissonRegression(step_size=lr)
    p.fit(x_train, y_train)
    prediction = p.predict(x_eval)
    np.savetxt(save_path, prediction)
    
    plt.figure()
    plt.xlim(0, 30)
    plt.ylim(0, 30)
    plt.xlabel('true count')
    plt.ylabel('predicted expected count')
    plt.scatter(y_eval, prediction)
    plt.savefig("poisson.png")
    
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to save_path
    # *** END CODE HERE ***


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        r = x.shape[0]
        c = x.shape[1]
        print(x[0])
        print(y[0])
        if self.theta == None:
            self.theta = np.zeros(c)

        for iter in range(self.max_iter):
            prevtheta = self.theta.copy()
            gradient = np.zeros(c)

            for i in range(r):
                gradient += (y[i] - np.exp(x[i].dot(self.theta)))*x[i]
                
            self.theta += self.step_size*gradient
            
            if np.sum(np.abs(self.theta - prevtheta)) < self.eps:
                break
        
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        return np.exp(x.dot(self.theta))
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
