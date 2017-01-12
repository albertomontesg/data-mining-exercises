import matplotlib.pyplot as plt
import numpy as np
from optparse import OptionParser
import sys


class cosine_hash:
    def __init__(self, kind, n):
        if kind == 'sketch':
            self.random_vec = np.random.choice([-1, 1], size=[1, n])
        else:
            self.random_vec = np.random.randn(1, n)

    def hash(self, vector):
        return np.sign(np.inner(vector, self.random_vec))


def plot_graph(options):
    # First row = true similarity, second row = probability they agree.
    results = np.zeros([2, options.n])
    hashes = [cosine_hash(options.hash, options.d) for _ in range(options.m)]
    for i in range(options.n):
        if options.sampling == 'uniform':
            # Sample uniformly from [-1,1].
            x = np.random.rand(1, options.d)*2 - 1
            y = np.random.rand(1, options.d)*2 - 1
        else:
            # Sample from a normal distribution.
            x = np.random.randn(1, options.d)
            y = np.random.randn(1, options.d)
        matches = 0
        for h in hashes:
            if h.hash(x) == h.hash(y):
                matches += 1
        results[0, i] = np.arccos(
            np.inner(x, y) / (np.linalg.norm(x)*np.linalg.norm(y)))
        results[1, i] = float(matches)/len(hashes)

    plt.plot(results[0, :], results[1, :], 'o')
    plt.xlim([0, np.pi])
    plt.ylim([0, 1])
    plt.xlabel('Angle between points')
    plt.ylabel('Probability of agreement')
    plt.title('%d pairs, %d hashes, %d dimensions' %
              (options.n, options.m, options.d))
    plt.rc('font', size=18)
    plt.show()

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--hash", type="choice", choices=("sketch", "normal"),
                      default="sketch")
    parser.add_option("--sampling", type="choice",
                      choices=("uniform", "normal"), default="uniform")
    parser.add_option("--n", type="int", default=1000, help="number of pairs")
    parser.add_option("--m", type="int", default=50, help="number of hashes")
    parser.add_option("--d", type="int", default=10, help="dimension")
    (options, args) = parser.parse_args()
    if options.n < 1 or options.m < 1 or options.d < 1:
        sys.stderr.write("The constants must be at least 1\n")
        sys.stderr.flush()
        sys.exit(1)

    plot_graph(options)
