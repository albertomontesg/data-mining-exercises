"""
Data Mining, FS 2015

Solution of Series 3 (Online Convex Programming)

For any questions, contact Alkis Gotovos <alkisg@inf.ethz.ch>
"""
import numpy as np
import scipy
import sklearn
import sklearn.metrics as skmet
import sklearn.grid_search as skgs
import sklearn.base as skbase
import sklearn.preprocessing as skpre
import matplotlib.pyplot as plt


def project_L2(w, a):
    """Project to L2-ball, as presented in the lecture."""
    return w * min(1, 1 / (np.sqrt(a) * np.linalg.norm(w, 2)))


def project_L1(w, a):
    """Project to L1-ball, as described by Duchi et al. [ICML '08]."""
    z = 1.0 / (a * a)
    if np.linalg.norm(w, 1) <= z:
        return w
    mu = -np.sort(-w)
    cs = np.cumsum(mu)
    rho = -1
    for j in range(len(w)):
        if mu[j] - (1.0 / (j + 1)) * (cs[j] - z) > 0:
            rho = j
    theta = (1.0 / (rho + 1)) * (cs[rho] - z)
    return np.sign(w) * np.fmax(w - theta, 0)


class OcpBase(skbase.BaseEstimator):
    """Base class for online convex programming."""
    def __init__(self, a=1.0):
        self.a = a

    def fit(self, x, y):
        raise NotImplementedError('Implement this in subclasses.')

    def predict(self, x):
        signs = np.sign(np.dot(x, self.w))
        signs[signs == 0] = -1
        return signs.astype('int8')


class OcpSvm(OcpBase):
    """Online SVM with L2 regularizer."""
    def fit(self, x, y):
        assert x.shape[0] == y.shape[0]
        w = np.zeros(x.shape[1])
        for t in range(x.shape[0]):
            if y[t] * np.dot(w, x[t, :]) < 1:
                eta = 1 / np.sqrt(t + 1)
                w += eta * y[t] * x[t, :]
                w = project_L2(w, self.a)
        self.w = w

    def __str__(self):
        return "SVM"


class OcpLogistic(OcpBase):
    """Online logistic regression with L1 regularizer."""
    def fit(self, x, y):
        assert x.shape[0] == y.shape[0]
        w = np.zeros(x.shape[1])
        for t in range(x.shape[0]):
            eta = 1 / np.sqrt(t + 1)
            ct = np.exp(-scipy.misc.logsumexp([0, y[t] * np.dot(w, x[t, :])]))
            w += eta * ct * y[t] * x[t, :]
            w = project_L1(w, self.a)
        self.w = w

    def __str__(self):
        return "Logistic"


def permute_data(x, y):
    perm = np.random.permutation(x.shape[0])
    return x[perm, :], y[perm]


def evaluate(model, data, nreps):
    Xtrain, Ytrain, Xtest, Ytest = data
    # Perform grid search with cross-validation to find a suitable regularizer
    print 'Performing grid search...'
    Xtrain, Ytrain = permute_data(Xtrain, Ytrain)
    pgrid = {'a': [0.001, 0.05, 0.01, 0.05, 0.1]}
    gs = skgs.GridSearchCV(model(), param_grid=pgrid, scoring='accuracy', cv=10)
    gs.fit(Xtrain, Ytrain)
    print 'Best regularizer found:', gs.best_params_['a']
    # For different sizes of training data, learn the model, predict on the
    # test data, and compute the prediction accuracy.
    # NOTE: We repeat this process `nreps` times for different permutations
    # of the data instances.
    print 'Training and evaluating...'
    training_sizes = list(
        np.round((np.logspace(0, np.log10(Xtrain.shape[0]), 15))))
    accuracy = np.zeros((nreps, len(training_sizes)))
    for rep in range(nreps):
        Xtrain, Ytrain = permute_data(Xtrain, Ytrain)
        ocp = model(a=gs.best_params_['a'])
        for i, size in enumerate(training_sizes):
            ocp.fit(Xtrain[:size, :], Ytrain[:size])
            Ypred = ocp.predict(Xtest)
            accuracy[rep, i] = skmet.accuracy_score(Ytest, Ypred)
    label = str(ocp)
    return label, training_sizes, accuracy


def read_data():
    print 'Reading data...'
    Xtrain = np.genfromtxt('data/Xtrain.csv', delimiter=',')
    Ytrain = np.genfromtxt('data/Ytrain.csv', delimiter=',', dtype='int8')
    Xtest = np.genfromtxt('data/Xtest.csv', delimiter=',')
    Ytest = np.genfromtxt('data/Ytest.csv', delimiter=',', dtype='int8')
    return Xtrain, Ytrain, Xtest, Ytest


def plot_accuracy(accs):
    for label, sizes, acc in results:
        acc_mean = np.mean(acc, axis=0)
        acc_std = 2 * np.std(acc, axis=0) / np.sqrt(nreps)
        plt.errorbar(sizes, acc_mean, yerr=acc_std, fmt='o-', linewidth=5,
                     elinewidth=2, capthick=2, label=label)
    plt.gca().set_xscale('log')
    plt.ylim((0.5, 1))
    plt.xlabel('Number of training instances')
    plt.ylabel('Test set accuracy')
    plt.legend(loc='lower right')
    plt.rcParams.update({'font.size': 20})
    plt.show()


if __name__ == '__main__':
    data = read_data()
    nreps = 100
    results = [evaluate(OcpSvm, data, nreps),
               evaluate(OcpLogistic, data, nreps)]
    plot_accuracy(results)
