import numpy as np


class BrainData:

    def __init__(self, brain, mask, stat):
        self.brain = brain
        self.mask = mask
        self.stat = stat

    def flatten(self):
        self.brain = np.reshape(self.brain, (-1))
        self.mask = np.reshape(self.mask, (-1))

    def flatten_masked(self):
        self.brain = np.reshape(self.brain[self.mask == True or self.mask == 1], (-1))

    def masking(self):
        return self.brain[self.mask == True or self.mask == 1]

