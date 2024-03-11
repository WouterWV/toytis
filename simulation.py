import logging
import numpy as np
from ensemble import Ensemble
from moves import shooting_move, swap, swap_zero, repptis_swap
from snakemove import snake_move, forced_extension
import pickle as pkl
from funcs import plot_paths
import matplotlib.pyplot as plt

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
        self.p_shoot = settings.get("p_shoot", 0.9)
        self.include_stateB = settings.get("include_stateB", False)
        self.prime_both_starts = settings.get("prime_both_starts", False)
        #self.snake_Lmax = settings.get("snake_Lmax", 5)
        #self.save_pe2 = settings.get("save_pe2", False)
        
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
        ens_set["simtype"] = self.simtype
        ens_set["temperature"] = self.settings["temperature"]
        ens_set["friction"] = self.settings["friction"]
        ens_set["dt"] = self.settings["dt"]
        ens_set["prime_both_starts"] = self.prime_both_starts
        # ens_set["save_pe2"] = self.save_pe2
        # ens_set["pe2_N"] = self.settings["pe2_N"]
        ens_set["max_paths"] = self.settings["max_paths"]

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

        # Dealing with state B ensembles, if requested
        if self.include_stateB:
            logger.info("Making state B ensembles")
            # Create the N+-' ensemble
            msg = "State B ensembles only implemented for repptis"
            assert self.simtype == "repptis", msg
            ens_set["id"] = len(self.intfs)
            ens_set["intfs"] = {"L": self.intfs[-2],
                                "M": None,  # Not even defined for N+-'
                                "R": self.intfs[-1]}
            ens_set["ens_type"] = "PPTIS_Nplusmin_primed"
            ens_set["name"] = f"[{len(self.intfs)-1}+-']"
            logger.info("Making ensemble {}".format(ens_set["name"]))
            self.ensembles.append(Ensemble(ens_set))
            # Create the N- ensemble
            ens_set["id"] = len(self.intfs) + 1
            ens_set["intfs"] = {"L": self.intfs[-1],
                                "M": None,  # Not even defined for N-
                                "R": np.infty}
            ens_set["ens_type"] = "state_B"
            ens_set["name"] = f"[{len(self.intfs)-1}-]"
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
        p_shoot = self.p_shoot
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

    ############################################################
    # Old functions that allowed a snake to use its memory of prior MCMC paths.
    # Detailed balance, however, forces a memoryless existence onto any snake 
    # that desires to live. No worry buddy, I will remember you for you.
    ############################################################
    def do_snake_move(self, tail_id=None):
        """Spawns a snake in a random ensemble.
        
        """
        self.cycle += 1
        logger.info("Cycle {}: Performing a snake move.".format(self.cycle))
        snake_skeleton, snake_paths, visits =\
            snake_move(self.ensembles, Lmax=self.snake_Lmax,
                       simcycle=self.cycle,
                       tail_id=tail_id)
        # Calculate the amount of ensemble visits and how many unique ensembles 
        # were visited
        tot_visits = np.sum(visits)
        unique_visits = np.sum(visits > 0)
        logger.info("Snake made {} visits to {} unique ensembles.".format(
            tot_visits, unique_visits))
        # Updating the data happens within snake_move itself

    def do_multi_level_shooting_moves(self):
        """Perform shooting moves in all the ensembles, where the level at 
        which we shoot is chosen between 0 or 1.
        
        """
        self.cycle += 1
        for ens in self.ensembles:
            level = np.random.choice([0,1])
            status, trial = shooting_move(ens, level=level)
            logger.info("Shooting move in {} at level {} resulted in {}".format(
                ens.name, level, status))
            ens.update_data(status, trial, "sh", self.cycle, update_paths=False)
            # update the ensemble paths list. (If the trial path is accepted!!)
            #
            # If we shot from level 0, then the trial path is the new level 1 
            # path. The original level 1 path is now the level 0 path. The 
            # original level 0 path becomes the level 3 path. For the other
            # paths, level N becomes level N+1. 
            # We can accomplish this quickly by just inserting the trial path
            # # at position 1, and then flipping level 0 with level 2. 
            # if level == 0 and status == "ACC":
            #     ens.paths.insert(1, trial)
            #     ens.paths[0], ens.paths[2] = ens.paths[2], ens.paths[0]
            #     if len(ens.paths) > ens.max_paths:
            #         ens.paths.pop()
            # # If we shot from level 1, then the trial path is the new level 0
            # # path. All other paths are just shifted one level up.
            # # We can accomplish this quickly by just inserting the trial path
            # # at position 0. 
            # if level == 1 and status == "ACC":
            #     ens.paths.insert(0, trial)
            #     if len(ens.paths) > ens.max_paths:
            #         ens.paths.pop()
            # The above didn't work, so we'll make it easier: 
            # If we shot from level 0, then trial = new level 1, and old level 1
            # is now level 0. We first make two empty spots in the paths list, 
            # and fill them up. 
            if level == 0 and status == "ACC":
                ens.paths.insert(0, trial)
                ens.paths.insert(0, ens.paths[2])  # this was the old level 1
            if level == 1 and status == "ACC":
                ens.paths.insert(0, ens.paths[0])  # this was old level 0
                ens.paths.insert(0, trial)
            if status != 'ACC':
                ens.paths.insert(0, ens.paths[1])  # This was old level 1
                ens.paths.insert(0, ens.paths[1])  # This was old level 0
            if len(ens.paths) > ens.max_paths:
                while len(ens.paths) > ens.max_paths:
                    ens.paths.pop()
            
            # We need to update the pe2 file
            # if self.save_pe2:
            #     ens.write_to_pe2(gen="sh")

    def run_snake(self):
        p_shoot = self.p_shoot
        while self.cycle < self.max_cycles:
            logger.info("-" * 80)
            logger.info("Cycle {}".format(self.cycle))
            logger.info("-" * 80)
            if np.random.rand() < p_shoot:
                self.do_shooting_moves()
            else:
                self.do_snake_move()

    def do_forced_extension_move(self, idx=None):
        self.cycle += 1
        # choose a random ensemble to start from 
        if idx is None:
            idx = np.random.choice(self.ensembles).id
        logger.info("Forced extension move in {} ({})".format(
            self.ensembles[idx].name, idx))
        status, trial, newidx = forced_extension(self.ensembles, idx)
        logger.info("Forced ext from {} ({}) to {} ({}) resulted in {}".format(
            self.ensembles[idx].name, idx,
            self.ensembles[newidx].name, newidx,
            status))
        # Update the data. 
        # Sadly, we will waste some computational time to enforce detailed
        # balance. The ensemble which we extend from will remove its last path
        # and also the path_data for that path. The ensemble which we extend to
        # gains the newly generated path, whose path_data is added to the 
        # pathensemble.txt file. 
        self.ensembles[newidx].update_data(status, trial, 'fn', self.cycle)
        #self.ensembles[newidx].paths.insert(0, trial)
        self.ensembles[idx].jump_back(n=1)
        # That's it..
        
        # ## OLD UPDATE CODE ##
        # # update the data
        # # new ensemble gets the trial to pathensemble.txt, while the old 
        # # ensemble gets its second_last path to pathensemble.txt. We update the
        # # ensemble paths list for the new ensemble using update_data (as this 
        # # also manages the max length of the paths list)
        # self.ensembles[newidx].update_data(status, trial, 'fn', self.cycle)
        # self.ensembles[idx].update_data(status, self.ensembles[idx].paths[1],
        #                                 'fo', self.cycle, update_paths=False)
        # # manage the ensemble path list of the old ensemble
        # # for the old ensemble, we pop the first path, and as a safety measure
        # # we insert the lowest level path again (such that we never run out of
        # # paths) TODO: this may lead to biases in extreme cases?
        # self.ensembles[idx].paths.pop(0)
        # self.ensembles[idx].paths.append(self.ensembles[idx].paths[-1])
        # # If the trial path is not accepted (which should not happen, unless
        # # FTX or BTX occurs), we are screwed.
        # if status != "ACC":
        #     logger.warning("Forced extension move failed somehow...")
        # # If we use the pe2 file, we need to write the paths to it. 
        # if self.save_pe2:
        #     self.ensembles[newidx].write_to_pe2(gen="fn")
        #     self.ensembles[idx].write_to_pe2(gen="fo")
        # # Let's plot the level 0 and level 1 paths of both ensembles before and 
        # # after this move has been finished. 
        # # fig, ax = plt.subplots()
        # # l0paths = [self.ensembles[idx].paths[0], self.ensembles[newidx].paths[0]]
        # # l1paths = [self.ensembles[idx].paths[1], self.ensembles[newidx].paths[1]]
        # # l2paths = [self.ensembles[idx].paths[2], self.ensembles[newidx].paths[2]]
        # # plot_paths(l0paths, ax=ax, start_ids=[0,0], color="r")
        # # plot_paths(l1paths, ax=ax, start_ids=[1000,1000], color="g")
        # # plot_paths(l2paths, ax=ax, start_ids=[2000,2000], color="b")
        # # ax.set_title("After the forced extension...")
        # # fig.show()
        # # Let's do it again, but plot make two subplots, one for idx and one for head
        # # fig, (ax1, ax2) = plt.subplots(1,2)
        # # idxpaths = [self.ensembles[idx].paths[0], self.ensembles[idx].paths[1],
        # #             self.ensembles[idx].paths[2]]
        # # headpaths = [self.ensembles[newidx].paths[0],
        # #              self.ensembles[newidx].paths[1],
        # #              self.ensembles[newidx].paths[2]]
        # # plot_paths(idxpaths, ax=ax1, start_ids="staggered")
        # # plot_paths(headpaths, ax=ax2, start_ids="staggered")
        # # ax1.set_title("{}: After the forced extension...".format(
        # #     self.ensembles[idx].name))
        # # ax2.set_title("{}: After the forced extension...".format(
        # #     self.ensembles[newidx].name))
        # # fig.show()

    def run_force(self):
        p_shoot = self.p_shoot
        while self.cycle < self.max_cycles:
            logger.info("-" * 80)
            logger.info("Cycle {}".format(self.cycle))
            logger.info("-" * 80)
            if np.random.rand() < p_shoot:
                self.do_shooting_moves()
            else:
                self.do_forced_extension_move()