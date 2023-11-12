class Path:
    """ Representation of a path in PPTIS. We keep a global counter of the
    number of paths that have been created, so that each path has a unique
    identifier.

    Attributes
    ----------
    phasepoints : list of tuples (x, v) of floats
        List of phasepoints in the path
    orders : list of floats
        List of order parameters for each phasepoint
    path_id : int
        Unique identifier of the path
    ens_id : int
        Unique identifier of the ensemble to which the path belongs

    """
    path_counter = 0

    def __init__(self, phasepoints=None, orders = None, ens_id=None):
        """Initialize the Path object.

        Parameters
        ----------
        phasepoints : list
            List of phasepoints in the path
        ens_id : int
            Unique identifier of the ensemble to which the path belongs
        path_id : int
            Unique identifier of the path
        """
        self.phasepoints = phasepoints
        self.orders = orders
        self.ens_id = ens_id
        self.path_id = Path.path_counter
        Path.path_counter += 1

    def copy_path(self):
        """Returns a copy of the path. Changes made to this new path will not be
        reflected in the original path.
        """
        return Path(self.phasepoints.copy(), self.orders.copy(), self.ens_id)