import matplotlib.pyplot as plt
import numpy as np

def hist_incoming_conn(weights, bin_size, histtype, gaussian_fit, savefig):

        """Args:
        :param weights(array) - Connection weights
        :param bin_size(int) - Histogram bin size
        :param histtype(str) - Same as histtype matplotlib
        :param gaussian_fit(bool) - If true; returns the plot with gaussian fit for corresponding histogram
        :param savefig(bool) - If True plot will be saved as png file in the cwd

        Returns:
        plot object """

        # Plot the histogram of distribution of number of incoming connections in the network

        num_incoming_weights = np.sum(np.array(weights) > 0, axis=0)

        plt.figure(figsize=(12, 5))

        plt.title('Number of incoming connections')
        plt.xlabel('Number of connections')
        plt.ylabel('Count')
        plt.hist(num_incoming_weights, bins=bin_size, histtype=histtype)

        if gaussian_fit:

            # Empirical average and variance are computed
            avg = np.mean(num_incoming_weights)
            var = np.var(num_incoming_weights)
            # From hist plot above, it is clear that connection count follow gaussian distribution
            pdf_x = np.linspace(np.min(num_incoming_weights), np.max(num_incoming_weights), 100)
            pdf_y = 1.0 / np.sqrt(2 * np.pi * var) * np.exp(-0.5 * (pdf_x - avg) ** 2 / var)

            plt.plot(pdf_x, pdf_y, 'k--', label='Gaussian fit')
            plt.axvline(x=avg, color='r', linestyle='--', label='Mean')
            plt.legend()

        if savefig:
            plt.savefig('hist_incoming_conn')

        return plt.show()

def hist_outgoing_conn(weights, bin_size, histtype, gaussian_fit, savefig):

    """Args:
            :param weights(array) - Connection weights
            :param bin_size(int) - Histogram bin size
            :param histtype(str) - Same as histtype matplotlib
            :param gaussian_fit(bool) - If True; returns the plot with gaussian fit for corresponding histogram
            :param savefig(bool) - If True plot will be saved as png file in the cwd

            Returns:
            plot object """

    # Plot the histogram of distribution of number of incoming connections in the network

    num_outgoing_weights = np.sum(np.array(weights) > 0, axis=1)

    plt.figure(figsize=(12, 5))

    plt.hist(num_outgoing_weights, bins=bin_size, histtype=histtype)
    plt.title('Number of Outgoing connections')
    plt.xlabel('Number of connections')
    plt.ylabel('Count')

    if gaussian_fit:

        # Empirical average and variance are computed
        avg = np.mean(num_outgoing_weights)
        var = np.var(num_outgoing_weights)
        # From hist plot above, it is clear that connection count follow gaussian distribution
        pdf_x = np.linspace(np.min(num_outgoing_weights), np.max(num_outgoing_weights), 100)
        pdf_y = 1.0 / np.sqrt(2 * np.pi * var) * np.exp(-0.5 * (pdf_x - avg) ** 2 / var)

        plt.plot(pdf_x, pdf_y, 'k--', label='Gaussian fit')
        plt.axvline(x=avg, color='r', linestyle='--', label='Mean')
        plt.legend()

    if savefig:
        plt.savefig('hist_outgoing_conn')

    return plt.show()

def weight_distribution(weights, bin_size, savefig):
    """Args:
        :param weights (array) - Connection weights
        :param bin_size(int) - Spike train will be splited into bins of size bin_size
        :param savefig(bool) - If True, plot will be saved in the cwd

        Returns:
        plot object"""

    weights = weights[weights >= 0.01]  # Remove the weight values less than 0.01 # As reported in article SORN 2013
    y, x = np.histogram(weights, bins=bin_size)  # Create histogram with bin_size
    plt.clf()
    plt.scatter(x[:-1], y, s=2.0, c='black')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')

    if savefig:
        plt.savefig('weight distribution')

    return plt.show()