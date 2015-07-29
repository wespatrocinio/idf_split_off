from __future__ import division
from math import exp, log

import numbers

from settings import SIGMOID

class Sigmoidal:
    """
    The idea of this class is to split off the IDF values using a sigmoid function (also known as logistic function).
    This kind of function starts on zero and saturates at 1, so the final value will be between zero and the max idf
      value of your database
    """

    def __init__(self):
        self.strength = SIGMOID.get('strength')
        self.lower_bound = SIGMOID.get('lower_bound')
        self.shift = self._get_shift()

    def _get_shift(self):
        return log(1 / SIGMOID.get('shift_precision')) / self.strength

    def _get_weight(self, idf_value):
        return self.lower_bound + (1 - self.lower_bound) / (1 + exp(-(self.strength * idf_value - self.shift)))

    def split_idf(self, idf_value):
        try:
            if isinstance(idf_value, numbers.Number):
                return idf_value * self._get_weight(idf_value)
        except:
            raise BaseException
