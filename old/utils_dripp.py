import pandas as pd
import numpy as np
from pathlib import Path

from dripp.trunc_norm_kernel.optim import em_truncated_norm
from dripp.cdl.utils import get_activation, get_atoms_timestamps, \
    get_events_timestamps


def get_dripp_results(cdl_model, z_hat_, sfreq, events, event_id,
                      dripp_params, save_df=True, save_dir=Path('.')):
    """

    """
    # Duration of the experiment, in seconds
    T = z_hat_.shape[2] / sfreq
    # Transform atoms' activations
    acti = get_activation(model=cdl_model,
                          z_hat=z_hat_.copy(),
                          shift=True)
    atoms_tt = get_atoms_timestamps(acti=acti,
                                    sfreq=sfreq,
                                    threshold=dripp_params.pop('threshold'))
    event_tt = get_events_timestamps(events=events, sfreq=sfreq)
    # Save DriPP results in pandas.DataFrame
    df_dripp = pd.DataFrame(
        columns=['atom', 'baseline', 'alpha', 'm', 'sigma', 'lower', 'upper'])
    for kk, acti_tt in enumerate(atoms_tt):
        if len(acti_tt) == 0:
            if dripp_params['verbose']:
                print("no activation found for atom %i" % kk)
            new_row = {
                'atom': kk,
                'baseline': np.nan,
                'alpha': np.nan,
                'm': np.nan,
                'sigma': np.nan,
                'lower': dripp_params['lower'],
                'upper': dripp_params['upper']
            }
        else:
            # Run DriPP with one driver
            res_params, _, _ = em_truncated_norm(
                acti_tt, driver_tt=event_tt[event_id], T=T,
                **dripp_params)
            baseline_hat, alpha_hat, m_hat, sigma_hat = res_params
            new_row = {
                'atom': kk,
                'baseline': baseline_hat,
                'alpha': alpha_hat,
                'm': m_hat,
                'sigma': sigma_hat,
                'lower': dripp_params['lower'],
                'upper': dripp_params['upper']
            }
        df_dripp = df_dripp.append(new_row, ignore_index=True)

    if save_df:
        df_dripp.to_csv(save_dir / 'df_dripp.csv')

    return df_dripp
