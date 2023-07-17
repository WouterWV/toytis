import numpy as np
import logging
from potential import Potential

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class LangevinEngine:
    """ Representation of a Langevin engine in PPTIS

    Attributes
    ----------
    settings : dict
        Dictionary containing the settings of the engine
    dt : float
        Time step of the engine
    temperature : float
        Temperature of the engine
    friction : float
        Friction coefficient of the engine
    dim : int
        Dimension of the engine
    potential : :py:class:`Potential` object
        Potential of the engine
    phasepoint : tuple (x, v) of floats
        Phasepoint of the engine

    """
    def __init__(self, settings):
        self.settings = settings
        self.dt = self.settings["dt"]
        self.temperature = self.settings["temperature"]
        self.friction = self.settings["friction"]
        self.potential = Potential()
        self.phasepoint = None

    def step(self, ph=None):
        """Performs a single step of the Langevin engine.

        Parameters
        ----------
        ph : tuple (x, v) of floats
            Phasepoint at which to perform the step

        Returns
        -------
        ph : tuple (x, v) of floats
            Phasepoint after the step
        """
        if ph is None:
            ph = (self.phasepoint[0], self.phasepoint[1])
        x, v = ph
        # Calculate the potential and force at the current phasepoint
        _, force = self.potential.potential_and_force(ph)

        # Calculate the stochastic force based on the temperature and friction
        stochastic_force = \
            np.sqrt(2 * self.temperature * self.friction) * np.random.randn()

        # Update the velocity and position using the Langevin equation
        v = v + (force - self.friction * v) * self.dt +\
              stochastic_force * np.sqrt(self.dt)
        x = x + v * self.dt
        self.phasepoint = (x,v)
        return self.phasepoint

    def draw_velocities(self):
        """Draws velocities from the Maxwell-Boltzmann distribution.

        Returns
        -------
        v : float
            Velocity drawn from the Maxwell-Boltzmann distribution
        """
        return np.sqrt(self.temperature) * np.random.randn()

    def set_phasepoint(self, ph):
        """ Sets the phasepoint of the engine.

        Parameters
        ----------
        ph : tuple (x, v) of floats
            Phasepoint to set the engine to

        """
        self.phasepoint = ph
