"""
Utils scripts for
"""

from alphacsc import BatchCDL, GreedyCDL
from alphacsc.utils.signal import split_signal


def run_csc(X, **cdl_params):
    """

    Parameters
    ----------
    X : numpy.ndarray
        the data to run the CSC on

    cdl_params : dict
        dictionary of CSC parameters, such as 'n_atoms', 'n_times_atoms', etc.

    Returns
    -------
    cdl_model

    z_hat_

    """
    print('Computing CSC')

    cdl_params = dict(cdl_params)
    n_splits = cdl_params.pop('n_splits', 10)
    use_batch_cdl = cdl_params.pop('use_batch_cdl', False)
    if use_batch_cdl:
        cdl_model = BatchCDL(**cdl_params)
    else:
        cdl_model = GreedyCDL(**cdl_params)

    X_splits = split_signal(X, n_splits=n_splits, apply_window=True)

    # Fit the model and learn rank1 atoms
    print('Running CSC')
    cdl_model.fit(X_splits)

    z_hat_ = cdl_model.transform(X[None, :])
    return cdl_model, z_hat_
