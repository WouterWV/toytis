import logging
import numpy as np
from ensemble import Ensemble
from moves import shooting_move, swap, swap_zero, repptis_swap
import pickle as pkl

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class Simulation:
    """ (RE)PPTIS simulation

    Attributes
    ----------
    settings : dict
        Dictionary of simulation settings
    intfs : list of floats
        List of interfaces
    cycle : int
        Cycle number
    max_cycles : int
        Maximum number of cycles
    ensembles : list of :py:class:`Ensemble` objects
        List of ensembles
    simtype : str
        Type of simulation to perform. This is either "repptis" or "retis"
    method : str
        How is the simulation initialized? Either via loading of acceptable
        paths for each ensemble, or via a restart of a previous simulation.
    permeability: bool
        Whether we use a lambda_{-1} interface for the [0-] ensemble.
    zero_left: float
        The left interface of the [0-] ensemble. Only used for permeability.

    """
    def __init__(self, settings):
        """Initialize the Simulation object.

        Parameters
        ---------
        settings : dict
            Dictionary of settings

        """
        self.settings = settings
        self.intfs = settings["interfaces"]
        self.cycle = settings.get("cycle", 0)
        self.max_cycles = settings.get("max_cycles", 1000000)
        self.simtype = settings["simtype"]
        self.method = settings["method"]
        self.permeability = settings.get("permeability", False)
        self.zero_left = settings.get("zero_left", None)
        self.simtype = settings.get("simtype", "retis")

        logger.info("Initializing the {} simulation.".format(self.simtype))
        # Making the ensembles
        self.ensembles = []
        
        if self.method != "restart":
            # Create the ensembles
            logger.info("Making the ensembles")
            self.create_ensembles()
            logger.info("Done making the ensembles")

            # No snake moves allowed as we only have level 1 paths
            self.settings["Snakeable"] = False
            self.settings["Snakewait"] = 5
            logger.info("Creating dummy initial paths for the ensembles")
            for ens in self.ensembles:
                ens.create_initial_path()
            logger.info("Done creating dummy initial paths for the ensembles")
        else:
            # load the ensembles from restart pickles
            logger.info("Loading restart pickles for the ensembles")
            self.load_ensembles_from_restart()
            logger.info("Done loading restart pickles for ensembles")

    def do_shooting_moves(self):
        """Perform shooting moves in all the ensembles.

        """
        self.cycle += 1
        for ens in self.ensembles:
            status, trial = shooting_move(ens)
            logger.info("Shooting move in {} resulted in {}".format(
                ens.name, status))
            ens.update_data(status, trial, "sh", self.cycle)


    def do_swap_moves(self):
        """ Perform swap moves in all the ensembles, where we swap the
        ensembles with their next neighbours.

        Scheme 1:
        null move in ensemble 0 and swap 1 with 2, 2 with 3, etc. If there is
        an even number of ensembles, we also do a null move in the last ensemble

        Scheme 2:
        Swap 0 with 1, 1 with 2, etc. If there is an odd number of ensembles,
        we do a null move in the last ensemble

        """
        self.cycle += 1

        scheme = np.random.choice([1, 2])
        odd = False if len(self.ensembles) % 2 == 0 else True
        if scheme == 1:
            self.do_null_move(0, "00")
            if not odd:
                self.do_null_move(-1, "00")
            for i in range(1, len(self.ensembles) - 1, 2):
                self.do_swap_move(i)
        elif scheme == 2:
            if odd:
                self.do_null_move(-1, "00")
            for i in range(0, len(self.ensembles) - 1, 2):
                self.do_swap_move(i)


    def do_swap_move(self, i):
        """ Perform a swap move in ensemble i.

        Parameters
        ----------
        i : int
            Index of the ensemble in which to perform the swap move.

        """
        if i == 0:
            status, trial1, trial2 = swap_zero(self.ensembles)
            logger.info("Swap move {} <-> {} resulted in {}".format(
                self.ensembles[i].name, self.ensembles[i+1].name, status))
            self.ensembles[i].update_data(status, trial1, "s+", self.cycle)
            self.ensembles[i+1].update_data(status, trial2, "s-", self.cycle)
            return

        if self.simtype == "retis":
            status, trial1, trial2 = swap(self.ensembles, i)
            logger.info("Swap move {} <-> {} resulted in {}".format(
                self.ensembles[i].name, self.ensembles[i+1].name, status))
            self.ensembles[i].update_data(status, trial1, "s+", self.cycle)
            self.ensembles[i+1].update_data(status, trial2, "s-", self.cycle)

        elif self.simtype == "repptis":
            status, trial1, trial2 = repptis_swap(self.ensembles, i)
            logger.info("Swap move {} <-> {} resulted in {}".format(
                self.ensembles[i].name, self.ensembles[i+1].name, status))
            self.ensembles[i].update_data(status, trial1, "s+", self.cycle)
            self.ensembles[i+1].update_data(status, trial2, "s-", self.cycle)

    def do_null_move(self, i, gen="00"):
        """ Perform a null move in ensemble i.

        Parameters
        ----------
        i : int
            Index of the ensemble in which to perform the null move.

        """
        self.ensembles[i].update_data("ACC", self.ensembles[i].last_path,
                                      gen, self.cycle)

    def create_ensembles(self):
        """ Create all the ensembles for the simulation.

        """
        # First we make the zero minus ensemble
        ens_set = {}
        ens_set["id"] = 0
        ens_set["max_len"] = self.settings["max_len"]
        ens_set["max_paths"] = self.settings.get("max_paths", 100)
        ens_set["simtype"] = self.simtype
        ens_set["temperature"] = self.settings["temperature"]
        ens_set["friction"] = self.settings["friction"]
        ens_set["dt"] = self.settings["dt"]
        if self.permeability:
            assert self.zero_left is not None, "No zero_left for permeability"
            ens_set["intfs"] = {"L": self.zero_left,
                                "M": None,  # Not needed for permeability
                                "R": self.intfs[0]}
            ens_set["ens_type"] = "state_A_lambda_min_one"
            ens_set["name"] = "[0-']"
            logger.info("Making ensemble {}".format(ens_set["name"]))
        else:
            ens_set["intfs"] = {"L": -np.infty,
                                "M": None,  # Not even defined for 0-
                                "R": self.intfs[0]}
            ens_set["ens_type"] = "state_A"
            ens_set["name"] = "[0-]"
            logger.info("Making ensemble {}".format(ens_set["name"]))
        self.ensembles.append(Ensemble(ens_set))

        # Then we make the zero plus ensemble
        ens_set["id"] = 1
        ens_set["intfs"] = {"L": self.intfs[0],
                            "M": None,  # Not even defined for 0+ or 0+-'
                            "R": self.intfs[1]}
        if self.simtype == "repptis":
            ens_set["ens_type"] = "PPTIS_0plusmin_primed"
            ens_set["name"] = "[0+-']"
        elif self.simtype == "retis":
            ens_set["ens_type"] = "RETIS_0plus"
            ens_set["name"] = "[0+]"
        logger.info("Making ensemble {}".format(ens_set["name"]))
        self.ensembles.append(Ensemble(ens_set))

        # Then we make the body ensembles
        for i in range(0, len(self.intfs) - 2):
            ens_set["id"] = i + 2
            if self.simtype == "repptis":
                ens_set["intfs"] = {"L": self.intfs[i],
                                    "M": self.intfs[i + 1],
                                    "R": self.intfs[i + 2]}
                ens_set["ens_type"] = "body_PPTIS"
                ens_set["name"] = f"[{i+1}+-]"
            elif self.simtype == "retis":
                ens_set["intfs"] = {"L": self.intfs[0],
                                    "M": self.intfs[i+1],
                                    "R": self.intfs[-1]}
                ens_set["ens_type"] = "body_TIS"
                ens_set["name"] = f"[{i+1}+]"
            logger.info("Making ensemble {}".format(ens_set["name"]))
            self.ensembles.append(Ensemble(ens_set))

    def load_ensembles_from_restart(self):
        """ Load the restart pickles for each ensemble. """
        for i in range(len(self.interfaces)):
            self.ensembles.append(Ensemble.load_restart_pickle(i))

    def save_simulation(self, filename):
        """ Save the simulation to a pickle file. """
        with open(filename, "wb") as f:
            pkl.dump(self, f)

    def run(self):
        p_shoot = 0.9
        while self.cycle < self.max_cycles:
            logger.info("-" * 80)
            logger.info("Cycle {}".format(self.cycle))
            logger.info("-" * 80)
            if np.random.rand() < p_shoot:
                self.do_shooting_moves()
            else:
                self.do_swap_moves()

    @classmethod
    def load_simulation(cls, filename):
        """Load a simulation object from a pickle file.
        
        Parameters
        ----------
        filename : str
            Name of the pickle file to load.
        """
        with open(filename, "rb") as f:
            return pkl.load(f)