import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class Potential:
    def __init__(self):
        """Initialize the Potential object.

        This one is a dummy potenttial representing a double well with
        sigmoid modulated sine bumps.

        """
        self.a = 1.
        self.b = 3.5
        self.c = 0.
        self.d = .5
        self.p = .3

        # some extra constants to speed up the calculation
        self.a4 = 4.*self.a
        self.b2 = 2.*self.b
        self.exp_p15 = np.exp(15.*self.p)
        self.p3 = 3.*self.p
        self.pi2 = 2.*np.pi
        self.pi2divp = 2.*np.pi/self.p

    def potential_and_force(self, ph):
        """Returns the potential and force at phasepoint ph.

        Parameters
        ----------
        ph : tuple (x, v) of floats containing the position and velocity of the
             Langevin particle at phasepoint ph

        Returns
        -------
        pot : float
            Potential at phasepoint ph
        force : float
            Force at phasepoint ph

        """
        x = ph[0]
        doublewell = self.a*x**4 - self.b*(x - self.c)**2
        bump = self.d * np.sin(2. * np.pi * x / self.p)
        modulation = (1.-1. / (1 + np.exp(-5. * (np.abs(x) - 3.*self.p))))
        pot = doublewell + bump * modulation

        # force
        f_doublewell = -self.a4*x**3 + self.b2*(x - self.c)
        deriv_bump = self.d * np.cos(self.pi2 * x / self.p) * self.pi2divp
        deriv_modulation = \
            - 5. * np.sign(x) * np.exp(5.*(np.abs(x) + self.p3)) /\
            (np.exp(5.*np.abs(x)) + self.exp_p15)**2
        f_bump = -1. * deriv_bump * modulation - bump * deriv_modulation
        f = f_doublewell + f_bump

        return pot, f
    
    def plot_potential(self, ax):
        """Plots the potential.

        Parameters
        ----------
        ax : matplotlib axis object
            Axis on which to plot the potential

        """
        x = np.linspace(-1.5, 1.5, 1000)
        pot = np.zeros_like(x)
        for i in range(len(x)):
            pot[i], _ = self.potential_and_force((x[i], 0.))
        ax.plot(x, pot)
    
    def plot_force(self, ax):
        """Plots the force.

        Parameters
        ----------
        ax : matplotlib axis object
            Axis on which to plot the force

        """
        x = np.linspace(-1.5, 1.5, 1000)
        f = np.zeros_like(x)
        for i in range(len(x)):
            _, f[i] = self.potential_and_force((x[i], 0.))
        ax.plot(x, f)
