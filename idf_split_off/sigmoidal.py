from __future__ import division
from math import exp, log

import numbers

PRECISION_BASE = 1000

class Sigmoidal:
    """
    The idea of this class is to split off the IDF values using a sigmoid function (also known as logistic function).
    This kind of function starts on zero and saturates at 1, so the final value will be between zero and the max idf
      value of your database
    """

    def __init__(self, max_value, range, lower_bound=0.2):
        self.lower_bound = lower_bound
        self.shift = self._get_shift(range)
        self.strength = self._get_strength(range, max_value)
    
    def _get_strength(self, range, max_value):
        return (2 * log(PRECISION_BASE)) / ((1 - 2*range) * max_value)

    def _get_shift(self, range):
        return log(PRECISION_BASE) * ( 1 + ( 1 / ( 1 - 2 * range ) ) )

    def _get_weight(self, idf_value):
        return self.lower_bound + (1 - self.lower_bound) / (1 + exp(-(self.strength * idf_value - self.shift)))

    def split_idf(self, idf_value):
        try:
            if isinstance(idf_value, numbers.Number):
                return idf_value * self._get_weight(idf_value)
        except:
            raise BaseException