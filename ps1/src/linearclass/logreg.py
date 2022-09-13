import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept = True)
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    prediction = clf.predict(x_eval)
    np.savetxt(save_path, prediction)
    util.plot(x_eval, y_eval, clf.theta, train_path[0:3] + ".png")
    

    
    
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to save_path
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
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
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        
        r = x.shape[0]
        c = x.shape[1]
        H = np.zeros([c,c])
        z = np.zeros([r,1])
        h = np.zeros([r,1])
        grad = np.zeros([c,1])
        prevTheta = np.ones([c,1])
        
        if self.theta == None:
            self.theta = np.zeros([c,1])

        for iter in range(self.max_iter):

            prevTheta = self.theta
            for j in range(c):
                for i in range(r):
                    z[i] = np.dot(x[i],self.theta)
                    h[i] = 1/(1+np.exp(-z[i]))
                    grad[j] += (h[i]-y[i])*x[i][j]


            grad = grad/r

            for k in range(r):
                for i in range(c):
                    for j in range(c):
                
                        H[i][j] += x[k][i]*x[k][j]*h[k]*(1-h[k])

            H = H/r
            self.theta = self.theta - np.matmul(np.linalg.inv(H),grad)
            if np.sum(np.abs(self.theta-prevTheta)) < self.eps:
                break

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.
        
        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        z = np.dot(x, self.theta)
        h_x = 1/(1 + np.exp(-z))
        return h_x
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
