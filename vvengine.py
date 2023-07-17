import numpy as np
import logging
from potential import Potential

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class VelocityVerletEngine:
    """ Representation of a VelocityVerlet engine in PPTIS, performing
    NVE dynamics.

    Attributes
    ----------
    settings : dict
        Dictionary containing the settings of the engine
    dt : float
        Time step of the engine
    temperature : float
        Temperature of the engine
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
        self.mass = self.settings.get("mass", 1.)
        self.potential = Potential()
        self.phasepoint = None
        self.temperature = self.settings["temperature"]

    def step(self, ph=None):
        """Perform one integration step using VV

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

        # Update the velocity and position using the VV equation
        v += 0.5 * self.dt * force
        x += self.dt * v
        _, force = self.potential.potential_and_force((x, v))
        v += 0.5 * self.dt * force

        self.phasepoint = (x, v)
        return self.phasepoint



    def draw_velocities(self, ph=None):
        """Sets the velocity of the engine to a random velocity drawn from the
        Maxwell-Boltzmann distribution.

        Notes
        -----
        Still not correctly implemented TODO
        
        Returns
        -------
        v : float
            Velocity drawn from the Maxwell-Boltzmann distribution
        """
        # if ph is None:
        #     ph = (self.phasepoint[0], self.phasepoint[1])
        # x, v = ph
        # # draw velocities from the Maxwell-Boltzmann distribution
        # v_drawn = np.sqrt(self.temperature) * np.random.randn()
        # # But we are in NVE, so we need to rescale the velocities such taht 
        # # the total energy is conserved
        # pot_new, _ = self.potential.potential_and_force((x, v_drawn))
        # pot_old, _ = self.potential.potential_and_force((x, v))
        # # don't forget to incorporate the mass term
        # v_new = np.sqrt(2*(pot_old - pot_new)/self.mass + v**2)

        # return v_new
        # now just pass along maxwell-boltzmann draw
        return np.sqrt(self.temperature) * np.random.randn()
    
    def set_phasepoint(self, ph):
        """ Sets the phasepoint of the engine.

        Parameters
        ----------
        ph : tuple (x, v) of floats
            Phasepoint to set the engine to

        """
        self.phasepoint = ph
