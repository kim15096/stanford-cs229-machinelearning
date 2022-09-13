import numpy as np
import util
import sys

sys.path.append('../linearclass')

### NOTE : You need to complete p01b_logreg implementation first!

from logreg import LogisticRegression

lg = LogisticRegression()
lg2 = LogisticRegression()

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

    # *** START CODE HERE ***
    x_train, t_train = util.load_dataset(train_path, label_col = 't', add_intercept=True)
    x_train, y_train = util.load_dataset(train_path, add_intercept = True)
    x_test, t_test = util.load_dataset(test_path, label_col = 't', add_intercept = True)
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept = True)
    x_eval, t_eval = util.load_dataset(valid_path, label_col = 't', add_intercept = True)    
    # Part (a): Train and test on true labels
    lg.fit(x_train, t_train)
    predictionA = lg.predict(x_test)
    util.plot(x_test, t_test, lg.theta, "2a.png")
    np.savetxt(output_path_true, predictionA)
    # Make sure to save predicted probabilities to output_path_true using np.savetxt()
    # Part (b): Train on y-labels and test on true labels
    lg2.fit(x_train, y_train)
    predictionB = lg2.predict(x_test)
    util.plot(x_test, t_test, lg2.theta, "2b.png")
    np.savetxt(output_path_naive, predictionB)
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    # Part (f): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to output_path_adjusted
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    yPredict = clf.predict(x_eval)
    alpha = np.dot(yPredict.T, y_eval)/np.sum(yPredict)
    util.plot(x_test, t_test, clf.theta, "2f.png", correction = alpha)
    y_test_predicted = clf.predict(x_test)
    y_test_predicted = np.multiply(1/alpha, y_test_predicted)
    np.savetxt(output_path_adjusted, y_test_predicted)

    # *** END CODER HERE

    

if __name__ == '__main__':
    main(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.txt')
