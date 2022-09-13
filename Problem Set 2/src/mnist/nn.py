import numpy as np
import matplotlib.pyplot as plt
import argparse

def softmax(x):
    """
    Compute softmax function for a batch of input values. 
    The first dimension of the input corresponds to the batch size. The second dimension
    corresponds to every class in the output. When implementing softmax, you should be careful
    to only sum over the second dimension.

    Important Note: You must be careful to avoid overflow for this function. Functions
    like softmax have a tendency to overflow when very large numbers like e^10000 are computed.
    You will know that your function is overflow resistent when it can handle input like:
    np.array([[10000, 10010, 10]]) without issues.

    Args:
        x: A 2d numpy float array of shape batch_size x number_of_classes

    Returns:
        A 2d numpy float array containing the softmax results of shape batch_size x number_of_classes
    """
    # *** START CODE HERE ***
    x -= np.max(x,axis = 1)[:,np.newaxis]
    
    return np.exp(x)/np.sum(np.exp(x),axis=1)[:,np.newaxis]
    # *** END CODE HERE ***

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Args:
        x: A numpy float array

    Returns:
        A numpy float array containing the sigmoid results
    """
    # *** START CODE HERE ***
    sigmoid = 1/(1 + np.exp(-x))
    return sigmoid
    # *** END CODE HERE ***

def get_initial_params(input_size, num_hidden, num_output):
    """
    Compute the initial parameters for the neural network.

    This function should return a dictionary mapping parameter names to numpy arrays containing
    the initial values for those parameters.

    There should be four parameters for this model:
    W1 is the weight matrix for the hidden layer
    b1 is the bias vector for the hidden layer
    W2 is the weight matrix for the output layers
    b2 is the bias vector for the output layer

    As specified in the PDF, weight matricesshould be initialized with a random normal distribution.
    Bias vectors should be initialized with zero.
    
    Args:
        input_size: The size of the input data
        num_hidden: The number of hidden states
        num_output: The number of output classes
    
    Returns:
        A dict mapping parameter names to numpy arrays
    """

    # *** START CODE HERE ***
    dic = {}
    
    dic['W1'] = np.random.normal(size=(input_size, num_hidden))
    dic['W2'] = np.random.normal(size=(num_hidden, num_output))
    dic['b1'] = np.zeros(num_hidden)
    dic['b2'] = np.zeros(num_output)

    
    return dic
    # *** END CODE HERE ***

def forward_prop(data, labels, params): 
    """
    Implement the forward layer given the data, labels, and params.
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network

    Returns:
        A 3 element tuple containing:
            1. A numpy array of the activations (after the sigmoid) of the hidden layer
            2. A numpy array The output (after the softmax) of the output layer
            3. The average loss for these data elements
    """
    # *** START CODE HERE ***
    size = data.shape[0]

    hiddenLayer = sigmoid(data.dot(params['W1']) + params['b1'])
    outputLayer = softmax(hiddenLayer.dot(params['W2']) + params['b2'])
    
    loss = np.sum(-labels * np.log(outputLayer))/size

    
    return (hiddenLayer,outputLayer,loss)

    # *** END CODE HERE ***

def backward_prop(data, labels, params, forward_prop_func):
    """
    Implement the backward propegation gradient computation step for a neural network
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.
        
        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***

    return backward_prop_regularized(data, labels, params, forward_prop_func, 0)

    # *** END CODE HERE ***


def backward_prop_regularized(data, labels, params, forward_prop_func, reg):
    """
    Implement the backward propagation gradient computation step for a neural network
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above
        reg: The regularization strength (lambda)

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.
        
        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***
    hidden, output, loss = forward_prop_func(data, labels, params)

    size = data.shape[0]
    diff = output - labels

    dlW2 = hidden.T.dot(diff)/size + 2*reg*params['W2']
    
    dlb2 = np.sum(diff,axis=0)/size

    hiddenP = hidden*(1-hidden)

    diffDW2 = diff.dot(params['W2'].T)
    
    dlW1 = ((data.T)@(diffDW2 * hiddenP))/size
    
    dlW1 = dlW1 + 2*reg*params['W1']
    
    dlb1 = np.sum(diffDW2 * hiddenP, axis=0)/size

    dic = {'dW1': dlW1, 'dW2': dlW2, 'db1': dlb1, 'db2': dlb2}
          
    return dic

    # *** END CODE HERE ***

def gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, forward_prop_func, backward_prop_func):
    """
    Perform one epoch of gradient descent on the given training data using the provided learning rate.

    This code should update the parameters stored in params.
    It should not return anything

    Args:
        train_data: A numpy array containing the training data
        train_labels: A numpy array containing the training labels
        learning_rate: The learning rate
        batch_size: The amount of items to process in each batch
        params: A dict of parameter names to parameter values that should be updated.
        forward_prop_func: A function that follows the forward_prop API
        backward_prop_func: A function that follows the backwards_prop API

    Returns: This function returns nothing.
    """

    # *** START CODE HERE ***
    
    no = train_data.shape[0]
    
    size = int(no/batch_size)
    
    for i in range(size):

        dic = backward_prop_func(train_data[i*1000:(i+1)*1000,:], train_labels[i*1000:(i+1)*1000,:], params, forward_prop_func)
        
        params['W1'] -= learning_rate*dic['dW1']
        params['W2'] -= learning_rate*dic['dW2']
        params['b1'] -= learning_rate*dic['db1']
        params['b2'] -= learning_rate*dic['db2']

    #*** END CODE HERE ***

    # This function does not return anything
    
    return

def nn_train(
    train_data, train_labels, dev_data, dev_labels, 
    get_initial_params_func, forward_prop_func, backward_prop_func,
    num_hidden=300, learning_rate=5, num_epochs=20, batch_size=1000):

    (nexp, dim) = train_data.shape

    params = get_initial_params_func(dim, num_hidden, 10)

    cost_train = []
    cost_dev = []
    accuracy_train = []
    accuracy_dev = []
    for epoch in range(num_epochs):
        gradient_descent_epoch(train_data, train_labels, 
            learning_rate, batch_size, params, forward_prop_func, backward_prop_func)

        h, output, cost = forward_prop_func(train_data, train_labels, params)
        cost_train.append(cost)
        accuracy_train.append(compute_accuracy(output,train_labels))
        h, output, cost = forward_prop_func(dev_data, dev_labels, params)
        cost_dev.append(cost)
        accuracy_dev.append(compute_accuracy(output, dev_labels))

    return params, cost_train, cost_dev, accuracy_train, accuracy_dev

def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) == 
        np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

def read_data(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y

def run_train_test(name, all_data, all_labels, backward_prop_func, num_epochs):
    params, cost_train, cost_dev, accuracy_train, accuracy_dev = nn_train(
        all_data['train'], all_labels['train'], 
        all_data['dev'], all_labels['dev'],
        get_initial_params, forward_prop, backward_prop_func,
        num_hidden=300, learning_rate=5, num_epochs=num_epochs, batch_size=1000
    )

    t = np.arange(num_epochs)

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(t, cost_train,'r', label='train')
    ax1.plot(t, cost_dev, 'b', label='dev')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.set_title('With Regularization')
    ax1.legend()

    ax2.plot(t, accuracy_train,'r', label='train')
    ax2.plot(t, accuracy_dev, 'b', label='dev')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('accuracy')
    ax2.legend()

    fig.savefig('./' + name + '.png')

    accuracy = nn_test(all_data['test'], all_labels['test'], params)
    print('For model %s, got accuracy: %f' % (name, accuracy))

def main():
    parser = argparse.ArgumentParser(description='Train a nn model.')
    parser.add_argument('--num_epochs', type=int, default=30)

    args = parser.parse_args()

    use_small_dataset = False  # make sure to turn this False for full training dataset
    
    #code taken from piazza to check on small datasets
    if use_small_dataset:
        print("Use small data set, update this once data loading and initial testing is done!")
        np.random.seed(100)
        train_data, train_labels = read_data('./images_train_small.csv', './labels_train_small.csv')
        train_labels = one_hot_labels(train_labels)
        p = np.random.permutation(4999)
        train_data = train_data[p,:]
        train_labels = train_labels[p,:]

        dev_data = train_data[0:1000,:]
        dev_labels = train_labels[0:1000,:]
        train_data = train_data[1000:,:]
        train_labels = train_labels[1000:,:]

    else:
        np.random.seed(100)
        train_data, train_labels = read_data('./images_train.csv', './labels_train.csv')
        train_labels = one_hot_labels(train_labels)
        p = np.random.permutation(60000)
        train_data = train_data[p,:]
        train_labels = train_labels[p,:]

        dev_data = train_data[0:10000,:]
        dev_labels = train_labels[0:10000,:]
        train_data = train_data[10000:,:]
        train_labels = train_labels[10000:,:]

    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    dev_data = (dev_data - mean) / std

    test_data, test_labels = read_data('./images_test.csv', './labels_test.csv')
    test_labels = one_hot_labels(test_labels)
    test_data = (test_data - mean) / std

    all_data = {
        'train': train_data,
        'dev': dev_data,
        'test': test_data
    }

    all_labels = {
        'train': train_labels,
        'dev': dev_labels,
        'test': test_labels,
    }
    
    run_train_test('baseline', all_data, all_labels, backward_prop, args.num_epochs)
    run_train_test('regularized', all_data, all_labels, 
        lambda a, b, c, d: backward_prop_regularized(a, b, c, d, reg=0.0001),
        args.num_epochs)

if __name__ == '__main__':
    main()
