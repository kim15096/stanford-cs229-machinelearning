import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    
    # *** START CODE HERE ***
    clf = GDA()
    clf.fit(x_train, y_train)
    prediction = clf.predict(x_eval)
    np.savetxt(save_path, prediction)
    util.plot(x_eval, y_eval, clf.theta, train_path[0:3] + "_gda.png")
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to save_path
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
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
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        
        r = x.shape[0]
        c = x.shape[1]
        pCounter = 0
        nCounter = 0
        mu_0 = np.zeros([1,c])
        mu_1 = np.zeros([1,c])
        phi = 0
        sigma = np.zeros([c,c])
        
        
        if self.theta == None:
                self.theta = np.zeros([c+1,1])
                
        for i in range(r):
        # Find phi, mu_0, mu_1, and sigma
            if y[i] == 1:
                phi += 1
                mu_1 += x[i]
                pCounter += 1                
            else:
                mu_0 += x[i]
                nCounter += 1
        
        phi = phi/r
        mu_0 = mu_0/nCounter
        mu_1 = mu_1/pCounter

        for i in range(r):
        # Find phi, mu_0, mu_1, and sigma
            if y[i] == 1:
                sigma += np.matmul((x[i]-mu_1).T,(x[i]-mu_1))
            else:
                sigma += np.matmul((x[i]-mu_0).T,(x[i]-mu_0))
                
        sigma = sigma/r
        # Write theta in terms of the parameters

        sigmaInv = np.linalg.inv(sigma)
        selfTheta = (-0.5*(np.matmul(mu_0,sigmaInv.T)- np.matmul(mu_1,sigmaInv.T)+ np.matmul(mu_0,sigmaInv) - np.matmul(mu_1,sigmaInv)))
        theta1 = selfTheta[0][0]
        theta2 = selfTheta[0][1]
        #selfTheta2 = (-0.5*(np.matmul(mu_0,sigmaInv.T)- np.matmul(mu_1,sigmaInv.T)+ np.matmul(mu_0,sigmaInv) - np.matmul(mu_1,sigmaInv)))[1]
        #self.theta[2:3] = (-0.5*(np.matmul(mu_0,sigmaInv.T)- np.matmul(mu_1,sigmaInv.T)+ np.matmul(mu_0,sigmaInv) - np.matmul(mu_1,sigmaInv)))
        theta0 = -0.5*(np.matmul(mu_1, np.matmul(sigmaInv, mu_1.T)) - np.matmul(mu_0, np.matmul(sigmaInv, mu_0.T))) - np.log(1/phi-1)
        self.theta[0] = theta0
        self.theta[1] = theta1
        self.theta[2] = theta2
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        
        z = np.matmul(x, self.theta)
        h_x = 1/(1 + np.exp(-z))
        return h_x
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
