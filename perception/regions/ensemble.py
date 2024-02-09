import numpy as np

from perception.regions.utils import RegionalCompetency
from perception.regions.cropping import Cropping
from perception.regions.masking import Masking
from perception.regions.perturbation import Perturbation
from perception.regions.gradients import Gradients
from perception.regions.reconstruction import Reconstruction


class Ensemble(RegionalCompetency):
    def __init__(self, methods, weights):
        self.methods = methods
        self.weights = weights

    def get_regions(self, data):
        comps = []
        for method in self.methods:
            comps.append(method.get_regions(data))
        comps = np.stack(comps)
        weights = np.array(self.weights)[:, np.newaxis, np.newaxis]
        return np.sum(comps * weights, axis=0) / np.sum(weights)