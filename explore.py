import numpy as np

# [reference]: https://keras.io/examples/rl/ddpg_pendulum/#:~:text=Deep%20Deterministic%20Policy%20Gradient%20(DDPG)%20is%20a%20model%2Dfree,algorithm%20for%20learning%20continous%20actions.&text=It%20uses%20Experience%20Replay%20and,operate%20over%20continuous%20action%20spaces
class OUActionNoise:
    """
    To implement better exploration by the Actor network, we use noisy perturbations, 
    specifically an Ornstein-Uhlenbeck process for generating noise, as described in the paper.
    """
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)