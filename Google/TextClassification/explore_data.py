import numpy as np
import matplotlib.pyplot as plt

sample_texts = "Ten minutes worth of story stretched out into the better part of two hours. When nothing of any significance had happened at the halfway point I should have left."

def get_num_words_per_sample(sample_texts):
    """Returns the median number of words per sample given corpus.

        # Arguments
            sample_texts: list, sample texts.

        # Returns
            int, median number of words per sample.
        """
    num_words = [len(s.split()) for s in sample_texts]
    print(num_words)
    print(np.median(num_words))
    return np.median(num_words)


def plot_sample_length_distribution(sample_texts):
    """Plots the sample length distribution.

    # Arguments
        samples_texts: list, sample texts.
    """
    plt.hist([len(s) for s in sample_texts], 50)
    plt.xlabel('Length of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')
    plt.show()
