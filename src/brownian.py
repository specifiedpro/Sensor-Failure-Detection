# brownian.py
import numpy as np

class Brownian:
    """
    Class for generating Brownian-motion-like processes.
    """
    def __init__(self, x0=0):
        """
        :param x0: Initial value
        """
        self.x0 = float(x0)

    def gen_random_walk(self, n_step=100):
        """
        Generate motion by simple random walk.

        :param n_step: Number of steps
        :return: 1D np.array of length n_step
        """
        if n_step < 30:
            print("WARNING! The number of steps is small.")
        w = np.ones(n_step) * self.x0
        for i in range(1, n_step):
            yi = np.random.choice([1, -1])
            w[i] = w[i-1] + (yi / np.sqrt(n_step))
        return w

    def gen_normal(self, n_step=100):
        """
        Generate motion by drawing from the Normal distribution.

        :param n_step: Number of steps
        :return: 1D np.array of length n_step
        """
        if n_step < 30:
            print("WARNING! The number of steps is small.")
        w = np.ones(n_step) * self.x0
        for i in range(1, n_step):
            yi = np.random.normal()
            w[i] = w[i-1] + (yi / np.sqrt(n_step))
        return w

    def stock_price(self, s0=100, mu=0.2, sigma=0.68, deltaT=52, dt=0.1):
        """
        Models a stock price S(t) ~ Weiner process.

        :param s0: initial stock price
        :param mu: drift
        :param sigma: volatility
        :param deltaT: total time
        :param dt: step size
        :return: 1D np.array of simulated stock prices
        """
        n_step = int(deltaT / dt)
        time_vector = np.linspace(0, deltaT, num=n_step)
        stock_var = (mu - (sigma**2 / 2)) * time_vector
        # Force initial value to zero for the wiener process logic
        self.x0 = 0
        weiner_process = sigma * self.gen_normal(n_step)
        s = s0 * (np.exp(stock_var + weiner_process))
        return s
