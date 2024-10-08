import logging
import os

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def make_ens_dirs_and_files(id):
    """ Makes the directory for ensemble with id, and creates the
    pathensemble.txt and order.txt files within it.

    Parameters
    ----------
    id : int
        Unique identifier of the ensemble

    """
    id = str(id)
    # make directory for ensemble
    if not os.path.exists(id):
        os.makedirs(id)
        logger.info("Created directory for ensemble %s", id)
    # make pathensemble.txt file
    if not os.path.exists(os.path.join(id, "pathensemble.txt")):
        with open(os.path.join(id, "pathensemble.txt"), "w") as f:
            f.write("")
        logger.info("Made file {} for ensemble {}".format(
            os.path.join(id, "pathensemble.txt"), id))
    else:
        logger.info("File {} already exists for ensemble {}".format(
            os.path.join(id, "pathensemble.txt"), id))
    # make pathensembe2.txt file
    if not os.path.exists(os.path.join(id, "pathensemble2.txt")):
        with open(os.path.join(id, "pathensemble2.txt"), "w") as f:
            f.write("")
        logger.info("Made file {} for ensemble {}".format(
            os.path.join(id, "pathensemble2.txt"), id))
    else:
        logger.info("File {} already exists for ensemble {}".format(
            os.path.join(id, "pathensemble2.txt"), id))
    # make order.txt file
    if not os.path.exists(os.path.join(id, "order.txt")):
        with open(os.path.join(id, "order.txt"), "w") as f:
            f.write("")
        logger.info("Made file {} for ensemble {}".format(
            os.path.join(id, "order.txt"), id))
    else:
        logger.info("File {} already exists for ensemble {}".format(
            os.path.join(id, "order.txt"), id))