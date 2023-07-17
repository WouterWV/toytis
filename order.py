import numpy as np
import logging

from path import Path

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class OrderParameter:
    def __init__(self, params = None):
        """Initialize the Ensemble object.mro

        Parameters
        ----------
        params : dict
            Dictionary of parameters for the order parameter.

        """
        self.params = params

    def calculate(self, ph):
        """ Calculate the order parameter for a phasepoint.

        Parameters
        ----------
        ph : tuple
            The phasepoint for which the order parameter is calculated.

        Returns
        -------
        order : list of floats
            The order parameter for the phasepoint.
        """
        # Here, we just return the x coordinate of the phasepoint
        return [ph[0]]