import numpy as np
from matplotlib import pyplot as plt

sample_data = np.loadtxt('sample-data.txt')

def create_histogram(data, bins=50):
    hist, bins = np.histogram(sample_data, bins=bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width, color='r')
    return hist.max()

def create_gaussian_plot(min, max, hypothesis, factor):
    plot_range = np.arange(min, max, 0.001)
    for i in range(hypothesis.size):
        plt.plot(plot_range, factor*probability_density(plot_range, hypothesis[i]))

def probability_density(x, mean):
    pd = np.exp(-(np.square(x-mean)/2))# / np.sqrt(2*np.pi)
    return pd

def find_expecations(x, hypothesis):
    expectations = np.ndarray((hypothesis.size, x.size))
    probability_densities = np.ndarray(expectations.shape)
    for i in range(expectations.shape[0]):
        probability_densities[i] = probability_density(x, hypothesis[i])
    for j in range(expectations.shape[0]):
        expectations[j] = probability_densities[j] / probability_densities.sum(axis=0)
    return expectations

def maximization(x, expectations):
    return (expectations * x).sum(axis=1) / expectations.sum(axis=1)

current_hypothesis = np.array([0.2, 1.2])
k = current_hypothesis.size
expectations = np.ndarray((k, sample_data.size))
iterations = 10


for i in range(iterations):
    expectations = find_expecations(sample_data, current_hypothesis)
    current_hypothesis = maximization(sample_data, expectations)

print(current_hypothesis)
max_count = create_histogram(sample_data)
create_gaussian_plot(sample_data.min(), sample_data.max(), current_hypothesis, max_count)
plt.show()

