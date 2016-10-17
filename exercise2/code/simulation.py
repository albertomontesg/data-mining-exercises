import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

N = 10**4
M = 1000
d = 10

def initialize_vectors(*shape):
    return np.random.rand(*shape) * 2 - 1

def angle(u, v):
    return np.arccos(np.dot(u,v)/(norm(u)*norm(v)))

# Initialize w vectors of the hash function
w = initialize_vectors(M,d)
def hash_H(v):
    return np.sign(np.dot(w, v))


def simulate():
    # Create the N pairs of random vectors
    v = initialize_vectors(N,2,d)

    hash_collision = np.empty((N,))
    angle_uv = np.empty((N,))

    # Iterate along pairs and compute the hash and cosine distance
    for i in range(N):
        h1 = hash_H(v[i,0])
        h2 = hash_H(v[i,1])

        angle_uv[i] = angle(v[i,0], v[i,1])
        hash_collision[i] = np.sum(h1==h2) / M

    plt.figure()
    plt.scatter(angle_uv, hash_collision, s=1)
    plt.ylim([0., 1.])
    plt.xlim([0., np.pi])
    plt.xlabel('angle(u,v)')
    plt.ylabel('$Pr([h(u)=h(v)])$')
    plt.savefig('simulation_vs_angle_{}.png'.format(d), bbox_inches='tight')

    plt.figure()
    plt.scatter(1-angle_uv/np.pi, hash_collision, s=1)
    plt.plot(np.linspace(0,1,20), np.linspace(0,1,20), 'r')
    plt.ylim([0., 1.])
    plt.xlim([0., 1.])
    plt.xlabel('1 - angle(u,v) / $\pi$')
    plt.ylabel('$Pr([h(u)=h(v)])$')
    plt.savefig('simulation_vs_similarity_{}.png'.format(d), bbox_inches='tight')


if __name__ == '__main__':
    simulate()
