import matplotlib.pyplot as plt
import numpy as np
from progressbar import ProgressBar
from sklearn.model_selection import KFold
from sklearn.utils import shuffle


class OnlineSVM(object):

    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def fit(self, X, y):
        assert len(X.shape) == 2, 'X shape invalid'
        self.w = np.zeros((X.shape[1],))

        nb_samples = X.shape[0]

        for t in range(nb_samples):
            nhu = 1. / np.sqrt(t+1)
            if np.dot(y[t], np.dot(self.w, X[t])) < 1:
                self.w += nhu * np.dot(y[t], X[t])
                self.w *= min(1.,
                              1. / (np.sqrt(self.lambda_) *
                                    np.linalg.norm(self.w, 2)))

    def predict(self, X):
        return np.sign(np.dot(X, self.w))

class OnlineLogisticRegression(object):

    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def fit(self, X, y):
        assert len(X.shape) == 2, 'X shape invalid'
        self.w = np.zeros((X.shape[1],))

        nb_samples = X.shape[0]

        for t in range(nb_samples):
            nhu = 1. / np.sqrt(t+1)

            self.w += nhu * np.dot(y[t], X[t]) / (1 + np.exp(np.dot(y[t], np.dot(self.w, X[t]))))
            self.w *= min(1.,
                          1. / (np.sqrt(self.lambda_) * np.linalg.norm(self.w, 1)))

    def predict(self, X):
        return np.sign(np.dot(X, self.w))


def load_data():
    X_train = np.genfromtxt('../data/Xtrain.csv', delimiter=',', dtype='float32')
    X_test = np.genfromtxt('../data/Xtest.csv', delimiter=',', dtype='float32')
    Y_train = np.genfromtxt('../data/Ytrain.csv', delimiter=',', dtype='float32')
    Y_test = np.genfromtxt('../data/Ytest.csv', delimiter=',', dtype='float32')
    X_train, Y_train = shuffle(X_train, Y_train, random_state=23)
    X_test, Y_test = shuffle(X_test, Y_test, random_state=23)
    return X_train, Y_train, X_test, Y_test


def compute_accuracy(y, y_pred):
    total = y.shape[0]
    return float(np.sum(y==y_pred)) / total

def train(Model, lambda_):
    # Load data
    X_train, Y_train, X_test, Y_test = load_data()

    model = Model(lambda_)
    model.fit(X_train, Y_train)

    y_test = model.predict(X_test)
    accuracy = compute_accuracy(Y_test, y_test) * 100.
    print('Accuracy: {:.2f}%'.format(accuracy))

def cv_lambda(Model, experiment=''):
    # Load data
    X_train, Y_train, X_test, Y_test = load_data()

    nb_folds = 10

    scores = list()

    lambdas = np.logspace(-4, 2, 30)
    for l in lambdas:
        model = Model(l)

        kf = KFold(n_splits=nb_folds, random_state=23)

        score = list()
        for train_idx, test_idx in kf.split(X_train):
            x_train = X_train[train_idx]
            y_train = Y_train[train_idx]
            x_test = X_train[test_idx]
            y_test = Y_train[test_idx]

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            score.append(compute_accuracy(y_test, y_pred))

        scores.append(score)

    scores = np.array(scores)

    scores_mean = scores.mean(axis=1)
    scores_std = scores.std(axis=1)

    plt.figure()
    plt.semilogx(lambdas, scores_mean)
    scores_error = scores_std / np.sqrt(nb_folds)

    plt.semilogx(lambdas, scores_mean + scores_error, 'b--')
    plt.semilogx(lambdas, scores_mean - scores_error, 'b--')
    plt.fill_between(lambdas, scores_mean + scores_error, scores_mean - scores_error, alpha=0.2)
    plt.ylabel('CV Accuracy +/- error')
    plt.xlabel('Lambda')
    plt.axhline(np.max(scores_mean), linestyle='--', color='.5')
    plt.xlim(lambdas[0], lambdas[-1])
    plt.savefig('../report/img/cross_val_{}.png'.format(experiment), bbox_inches='tight')

def classifcation_vs_training_samples(Model, lambda_, experiment=''):
    np.random.seed(23)
    # Load data
    X_train, Y_train, X_test, Y_test = load_data()

    total_nb_samples = X_train.shape[0]

    accuracies = list()
    nb_samples = np.linspace(0, total_nb_samples, 100)
    for n in nb_samples:
        permutation = np.random.permutation(total_nb_samples)
        idx = permutation[:int(n)]
        x_train, y_train = X_train[idx], Y_train[idx]

        model = Model(lambda_)
        model.fit(x_train, y_train)
        Y_pred = model.predict(X_test)
        accuracies.append(compute_accuracy(Y_test, Y_pred))

    class_error = 1 - np.array(accuracies)

    plt.figure()
    plt.plot(nb_samples, class_error)
    plt.title('Accuracy vs number of samples used at training')
    plt.xlabel('Number of samples for training')
    plt.ylabel('Classification Error')
    plt.xlim(nb_samples[0], nb_samples[-1])
    plt.ylim(0, 1)
    plt.savefig('../report/img/acc_vs_samples_{}.png'.format(experiment), bbox_inches='tight')
    plt.xlim(nb_samples[0], nb_samples[8])
    plt.savefig('../report/img/acc_vs_samples_zoom_{}.png'.format(experiment), bbox_inches='tight')

if __name__ == '__main__':
    # OnlineSVM
    Model = OnlineSVM
    cv_lambda(Model, experiment='svm')
    classifcation_vs_training_samples(Model, lambda_=.01, experiment='svm')
    train(Model, lambda_=.01)

    # OnlineLogisticRegression
    Model = OnlineLogisticRegression
    cv_lambda(Model, experiment='logistic')
    classifcation_vs_training_samples(Model, lambda_=.0001, experiment='logistic')
    train(Model, lambda_=.0001)
