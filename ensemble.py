import logging
import numpy as np
from path import Path
from filehandler import make_ens_dirs_and_files
from engine import LangevinEngine
from order import OrderParameter
import pickle as pkl

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

PATH_FMT = (
    '{0:>10d} {1:>10d} {2:>10d} {3:1s} {4:1s} {5:1s} {6:>7d} '
    '{7:3s} {8:2s} {9:>16.9e} {10:>16.9e} {11:>7d} {12:>7d} '
    '{13:>16.9e} {14:>7d} {15:7d} {16:>16.9e}'
)


class Ensemble:
    """ Representation of an ensemble in PPTIS

    Attributes
    ----------

    id : int
        Unique identifier of the ensemble
    intfs : dict of floats
        Dictionary of the L M R interfaces of the ensemble
    paths : list of :py:class:`Path` objects
        List of previously accepted paths in the ensemble
    data : list
        List containing data on the paths generated for each cycle in the
        ensemble
    max_len : int
        Maximum length of a path in the ensemble
    settings : dict
        Dictionary containing the settings of the ensemble
    start_conditions : set of str
        Set containing the start conditions of the ensemble
    end_conditions : set of str
        Set containing the end conditions of the ensemble
    cross_conditions : set of str
        Set containing the cross conditions of the ensemble
    ens_type : str
        String indicating the type of ensemble
    engine : :py:class:`LangevinEngine` object
        Engine of the ensemble
    paths : list of :py:class:`Path` objects
        List of paths in the ensemble
    max_paths : int
        Maximum number of paths that are kept in memory for the ensemble
    cycle : int
        Cycle number of the ensemble, i.e. the number of paths that have been
        generated for the ensemble (including rejected paths).
    acc_cycle : int
        How many cycles resulted in an accepted path?
    md_cycle : int
        How many cycles included the MD engine to generate the accepted paths?
    last_path : :py:class:`Path` object
        Last path that was **accepted** for the ensemble
    name : str
        Name of the ensemble
    order : :py:class:`OrderParameter` object
        Order parameter of the ensemble
    extremal_conditions : set of str
        Set containing the extremal conditions of the ensemble. This is the
        union of the start and end conditions.
    simtype : str
        Type of simulation to perform. This is either "repptis" or "retis"

    """

    def __init__(self, settings):
        """Initialize the Ensemble object.

        Parameters
        ----------
        settings : dict
            Dictionary containing the settings of the simulation
        """

        self.id = settings["id"]
        make_ens_dirs_and_files(self.id)
        self.settings = settings
        self.intfs = self.settings["intfs"]
        self.max_len = self.settings["max_len"]
        self.ens_type = self.settings["ens_type"]
        self.paths = []
        self.data = []
        self.max_paths = self.settings["max_paths"]
        self.name = settings.get("name", str(id))
        self.cycle = 0
        self.cycle_acc = 0
        self.cycle_md = 0
        self.simtype = settings["simtype"]

        # Set the start, end and cross conditions of the ensemble
        self.set_conditions()
        # extremal conditions is the union of start and end conditions
        self.extremal_conditions = self.start_conditions.union(
            self.end_conditions)

        # Set the engine of the ensemble
        self.set_engine()

        # Set the order parameter of the ensemble
        self.set_order_parameter()

    def set_engine(self):
        """ Sets the engine of the ensemble.

        """
        self.engine = LangevinEngine(self.settings)

    def set_order_parameter(self):
        self.orderparameter = OrderParameter(self.settings)

    def update_data(self, status, trial, gen, simcycle):
        """Updates the data of the path ensemble after a move has been
        performed. If the path is accepted, the last_path and paths attributes
        are updated.

        Parameters
        ----------
        status : str
            Status of the move. This is either "ACC" or any of the "REJ" flags.
        trial : :py:class:`Path` object
            Trial path
        gen : int
            Generation of the move: swap (s-, s+), shoot (sh), null (00)
        simcycle : int
            Cycle number of the simulation

        """
        # path data to obtain:
        ordermin = min([op[0] for op in trial.orders])
        ordermax = max([op[0] for op in trial.orders])
        ptype = self.get_ptype(trial)
        plen = len(trial.phasepoints)
        #self.paths.append(trial)
        self.cycle += 1
        if status == "ACC":
            self.cycle_acc += 1
            self.last_path = trial
            # insert the path at the beginning of the list
            self.paths.insert(0, trial)
            if len(self.paths) > self.max_paths:
                # remove the last path from the list
                self.paths.pop()
        if self.simtype == "retis":
            if gen == "sh":
                self.cycle_md += 1
        if self.simtype == "repptis":
            if gen in ["s-", "s+", "sh"]:
                self.cycle_md += 1

        # Now we write the data to the path ensemble file
        self.write_to_pe_file(simcycle, self.cycle_acc, self.cycle_md, ptype,
                              plen, status, gen, ordermin, ordermax)
        # and write to the order.txt file
        # self.write_to_order_file(trial, self.cycle, ptype, plen, status, gen)

    def write_to_pe_file(self, simcycle, cycle_acc, cycle_md, ptype, plen,
                         status, gen, ordermin, ordermax):
        """Format: simcycle, cycle_acc, cycle_md, ptype, plen, status,
        gen, ordermin, ordermax
        chars: 10, 10, 6, 7, 3, 2, 10decimals, 10decimals, 10
        align: right, right, right, right, right, right, right, right, right
        type: int, int, int, str, int, str, str, float, float

        """

        PATH_FMT = (
            '{0:>10d} {1:>10d} {2:>10d} {3:1s} {4:1s} {5:1s} {6:>7d} '
            '{7:3s} {8:2s} {9:>16.9e} {10:>16.9e} {11:>7d} {12:>7d} '
            '{13:>16.9e} {14:>7d} {15:7d} {16:>16.9e}'
        )
        with open(str(self.id) + "/pathensemble.txt", "a") as f:
            f.write(PATH_FMT.format(
                simcycle, cycle_acc, cycle_md, ptype[0], ptype[1], ptype[2],
                plen, status, gen,
                ordermin, ordermax, 0, 0, 0., 0, 0, 1.) + "\n")

    def set_conditions(self):
        """ Determines the start, end and cross conditions of the ensemble.

        Sets the attributes start_conditions, end_conditions and
        cross_conditions of the ensemble. These are sets containing one or more
        of the following strings: "L", "M" and "R".

        We distinguish between the following types of ensembles:
        - body_TIS: This is a regular [i^+] ensemble in (PP)TIS simulations.
        - state_A: This is the regular [0^-] ensemble in (PP)TIS simulations.
        - state_A_lambda_min_one: This is the [0^-'] ensemble in (PP)TIS
            simulations, where a lambda_{-1} is present.
        - body_PPTIS: This is a regular [i^{+-}] ensemble in PPTIS simulations.
        - PPTIS_0plusmin_primed: This is the [0^{+-}'] ensemble in PPTIS
            simulations.
        - state_B: This is the [N^-] ensemble in a snakeTIS simulation.
        - state_B_lambda_plus_one: This is the [N^-'] ensemble in a snakeTIS
            simulation, where a lambda_{N+1} is present.

        """

        if self.ens_type == "body_TIS":
            self.start_conditions = {"L"}
            self.end_conditions = {"L", "R"}
            self.cross_conditions = {"M"}

        elif self.ens_type == "state_A":
            self.start_conditions = {"R"}
            self.end_conditions = {"R"}
            self.cross_conditions = {}

        elif self.ens_type == "state_A_lambda_min_one":
            self.start_conditions = {"L", "R"}
            self.end_conditions = {"L", "R"}
            self.cross_conditions = {}

        elif self.ens_type == "body_PPTIS":
            self.start_conditions = {"L", "R"}
            self.end_conditions = {"L", "R"}
            self.cross_conditions = {"M"}

        elif self.ens_type == "PPTIS_0plusmin_primed":
            self.start_conditions = {"L"}  # no R because repptis_swap!!
            self.end_conditions = {"L", "R"}
            self.cross_conditions = {}

        elif self.ens_type == "RETIS_0plus":
            self.start_conditions = {"L"}
            self.end_conditions = {"L", "R"}
            self.cross_conditions = {}

        elif self.ens_type == "state_B":
            self.start_conditions = {"L"}
            self.end_conditions = {"L"}
            self.cross_conditions = {}

        elif self.ens_type == "state_B_lambda_plus_one":
            self.start_conditions = {"L", "R"}
            self.end_conditions = {"L", "R"}
            self.cross_conditions = {}

        else:
            raise ValueError("Unknown ensemble type: {}".format(self.ens_type))

    def check_ph_in_ensemble(self, ph):
        """ Checks whether a phasepoint could be valid for of a path in the
        pathensemble.

        Parameters
        ----------
        ph : tuple (x, v) of floats
            Phasepoint to check

        Returns
        -------
        bool
            True if the phasepoint could be valid for a path in the
            pathensemble, False otherwise

        """
        x = ph[0]
        if self.ens_type in \
            ["body_TIS", "body_PPTIS", "state_A_lambda_min_one",
             "state_B_lambda_plus_one", "PPTIS_0plusmin_primed"]:
            return x >= self.intfs["L"] and x <= self.intfs["R"]

        elif self.ens_type in ["state_A"]:
            return x <= self.intfs["L"]

        elif self.ens_type in ["state_B"]:
            return x >= self.intfs["R"]

        else:
            raise ValueError("Unknown ensemble type: {}".format(self.ens_type))

    def check_path(self, path):
        """ Checks whether a path is valid for the ensemble.

        Parameters
        ----------
        path : :py:class:`Path` object
            Path to check

        Returns
        -------
        bool
            True if the path is valid for the ensemble, False otherwise

        """
        return self.check_start(path) and self.check_end(path) and \
            self.check_cross(path)

    def check_cross(self, path):
        """ Checks whether a path meets the cross conditions of the ensemble.

        Parameters
        ----------
        path : :py:class:`Path` object
            Path to check

        Returns
        -------
        bool
            True if the path meets the cross conditions of the ensemble,
            False otherwise

        """
        crossed = True
        # if there are no cross conditions, automatically return True
        if len(self.cross_conditions) == 0:
            return True
        # if there are cross conditions, check whether the path meets them
        ordermax = max([op[0] for op in path.orders])
        ordermin = min([op[0] for op in path.orders])
        for cross_condition in self.cross_conditions:
            crossed = crossed and \
                (ordermax >= self.intfs[cross_condition] and \
                 ordermin <= self.intfs[cross_condition])
        return crossed

    def check_start_and_end(self, path):
        """ Checks whether a path meets the start and end conditions of the
        ensemble.

        Parameters
        ----------
        path : :py:class:`Path` object
            Path to check

        Returns
        -------
        bool
            True if the path meets the start and end conditions of the ensemble,
            False otherwise

        """
        return self.check_start(path) and self.check_end(path)

    def check_start(self, path):
        """ Checks whether a path meets the start conditions of the ensemble.

        Parameters
        ----------
        path : :py:class:`Path` object
            Path to check

        Returns
        -------
        bool
            True if the path meets the start conditions of the ensemble,
            False otherwise

        """
        start = False
        op = path.orders[0][0]
        if len(self.start_conditions) == 0:
            return True
        for start_condition in self.start_conditions:
            if start_condition == "R":
                start = start or op >= self.intfs[start_condition]
            elif start_condition == "L":
                start = start or op <= self.intfs[start_condition]
            else:
                raise ValueError("Unknown start condition: {}".format(
                    start_condition))
        return start

    def check_end(self, path):
        """ Checks whether a path meets the end conditions of the ensemble.

        Parameters
        ----------
        path : :py:class:`Path` object
            Path to check

        Returns
        -------
        bool
            True if the path meets the end conditions of the ensemble,
            False otherwise

        """
        end = False
        op = path.orders[-1][0]
        if len(self.end_conditions) == 0:
            return True
        for end_condition in self.end_conditions:
            if end_condition == "R":
                end = end or op >= self.intfs[end_condition]
            elif end_condition == "L":
                end = end or op <= self.intfs[end_condition]
            else:
                raise ValueError("Unknown end condition: {}".format(
                    end_condition))
        return end

    def check_start_end_positions(self, path):
        """ Returns the start and end positions of a path.
        These are either "L", "R", or "M" for left, right, or middle,
        respectively. R denotes right of the right interface, L denotes left of
        the left interface, and M denotes between the left and right interface.

        Parameters
        ----------
        path : :py:class:`Path` object
            Path to check

        Returns
        -------
        tuple of strings
            Start and end positions of the path

        """
        startpos = "L" if path.orders[0][0] <= self.intfs["L"] else \
            "R" if path.orders[0][0] >= self.intfs["R"] else "*"
        endpos = "L" if path.orders[-1][0] <= self.intfs["L"] else \
            "R" if path.orders[-1][0] >= self.intfs["R"] else "*"
        return startpos, endpos

    def get_ptype(self, path):
        """ Returns the type of the path. This is a three letter combination of
        L, M, R and * (for left, middle, right and unknown, respectively)

        Parameters
        ----------
        path : :py:class:`Path` object
            Path to check

        Returns
        -------
        str
            Type of the path

        """
        startpos, endpos = self.check_start_end_positions(path)
        middlepos = "M" if self.check_cross(path) else "*"
        return startpos + middlepos + endpos

    def create_initial_path(self, N = 250):
        """Create an initial path for the ensemble.
        We will create a path that starts at one of the start_condition intfs,
        and ends in one of the end_condition intfs. We check whether there is a
        crossing condition, and if so, make sure it is met.

        Parameters
        ----------
        N : int
            Half of the number of phasepoints in the initial path

        """
        logger.info("Creating initial path for ensemble {}".format(self.name))
        if self.ens_type == "body_TIS":
            # For body ensembles, we start at the left interface
            start = self.intfs["L"]*(1-np.sign(self.intfs["L"])*0.001)
            mid = self.intfs["M"]*(1+np.sign(self.intfs["M"])*0.001)
            stop = self.intfs["L"]*(1-np.sign(self.intfs["L"])*0.001)
        elif self.ens_type == "state_A":
            # For state A ensembles, we start at the right interface
            start = self.intfs["R"]*(1 + np.sign(self.intfs["R"])*0.001)
            mid = self.intfs["R"]*(1 - np.sign(self.intfs["R"])*0.1)
            stop = self.intfs["R"]*(1 + np.sign(self.intfs["R"])*0.001)
        elif self.ens_type == "state_A_lambda_min_one":
            # For state A ensembles, we start at the right interface
            start = self.intfs["R"]*(1 + np.sign(self.intfs["R"])*0.001)
            mid = (self.intfs["R"] + self.intfs["L"]) / 2
            stop = self.intfs["L"]*(1 - np.sign(self.intfs["L"])*0.001)
        elif self.ens_type == "body_PPTIS":
            # For body ensembles, we start at the left interface
            start = self.intfs["L"]*(1 - np.sign(self.intfs["L"])*0.001)
            mid = self.intfs["M"]
            stop = self.intfs["R"]*(1 + np.sign(self.intfs["R"])*0.001)
        elif self.ens_type == "PPTIS_0plusmin_primed":
            # For body ensembles, we start at the left interface
            start = self.intfs["L"]*(1 - np.sign(self.intfs["L"])*0.001)
            mid = (self.intfs["R"] + self.intfs["L"]) / 2
            stop = self.intfs["R"]*(1 + np.sign(self.intfs["R"])*0.001)
        elif self.ens_type == "RETIS_0plus":
            # For body ensembles, we start at the left interface
            start = self.intfs["L"]*(1 - np.sign(self.intfs["L"])*0.001)
            mid = (self.intfs["L"] + self.intfs["R"]) / 2
            stop = self.intfs["L"]*(1 - np.sign(self.intfs["L"])*0.001)
        elif self.ens_type == "state_B":
            raise NotImplementedError("State B ensembles not implemented yet")
        elif self.ens_type == "state_B_lambda_plus_one":
            raise NotImplementedError("State B ensembles not implemented yet")
        else:
            raise ValueError("Unknown ensemble type: {}".format(self.ens_type))
        # We make two subpaths, from start to mid, and from mid to stop
        # We set the velocity of each point to zero.
        phasepoints1 = [(i,0.) for i in np.linspace(start, mid, N)]
        phasepoints2 = [(i,0.) for i in np.linspace(mid, stop, N)]
        orders1 = [self.orderparameter.calculate(ph) for ph in phasepoints1]
        orders2 = [self.orderparameter.calculate(ph) for ph in phasepoints2]
        phasepoints = phasepoints1 + phasepoints2[1:]
        orders = orders1 + orders2[1:]
        path = Path(phasepoints, orders, self.id)
        self.paths.append(path)
        self.last_path = path
        self.update_data("ACC", path, "ld", 0)

    def write_restart_pickle(self):
        """ Writes a pickle file containing the ensemble data.
        
        """
        with open(str(self.id) + "/restart.pkl", "wb") as f:
            pkl.dump(self, f)

    @classmethod
    def load_restart_pickle(cls, id):
        """ Loads a pickle file containing the ensemble data.
        
        """
        with open(str(id) + "/restart.pkl", "rb") as f:
            return pkl.load(f)