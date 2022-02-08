"""For all subjects in Cam-CAN folder, run CSC and compute informations. Save
the final result in a dataframe and pickle.
"""

import numpy as np
import pandas as pd
import pickle
import json
from joblib import Memory, Parallel, delayed

from alphacsc.viz.epoch import make_epochs

from config import (BIDS_ROOT, RESULTS_DIR, PARTICIPANTS_FILE, N_JOBS,
                    CDL_PARAMS, EXP_PARAMS, get_cdl_pickle_name)
from utils_csc import get_raw, run_csc, get_subject_info, get_subject_dipole

mem = Memory('.')


def procedure(subject_id):
    """


    """
    # get preprocessed raw and events
    raw, events = get_raw(subject_id)
    # run multivariate CSC
    cdl_model, z_hat_ = mem.cache(run_csc)(
        X=raw.get_data(picks=['meg']), **CDL_PARAMS)
    # events here are only "good" button events
    events_no_first_samp = events.copy()
    events_no_first_samp[:, 0] -= raw.first_samp
    info = raw.info.copy()
    info.update(events=events_no_first_samp, event_id=None)

    allZ = make_epochs(
        z_hat_, info, t_lim=(-1.7, 1.7), n_times_atom=CDL_PARAMS['n_times_atom'])

    # save CDL results in pickle, as well as experiment parameters
    subject_res_dir = RESULTS_DIR / subject_id
    subject_res_dir.mkdir(parents=True, exist_ok=True)
    pickle.dump([cdl_model, raw.info, allZ, z_hat_],
                open(subject_res_dir / get_cdl_pickle_name(), "wb"))
    with open(subject_res_dir / 'exp_params', 'w') as fp:
        json.dump([EXP_PARAMS, CDL_PARAMS], fp, sort_keys=True, indent=4)

    # get informations about the subject
    age, sex, hand = get_subject_info(subject_id, PARTICIPANTS_FILE)
    base_row = {'subject_id': subject_id, 'age': age, 'sex': sex, 'hand': hand}
    # get informations about atoms
    dip = get_subject_dipole(subject_id, cdl_model, info=raw.info)

    new_rows = []
    for kk, (u, v) in enumerate(zip(cdl_model.u_hat_, cdl_model.v_hat_)):
        gof, pos, ori = dip.gof[kk], dip.pos[kk], dip.ori[kk]

        # calculate the percent change in activation between different phases of movement
        # -1.25 to -0.25 sec (150 samples)
        pre_sum = np.sum(allZ[:, kk, 68:218])
        # -0.25 to 0.25 sec (75 samples)
        move_sum = np.sum(allZ[:, kk, 218:293])
        # 0.25 to 1.25 sec (150 samples)
        post_sum = np.sum(allZ[:, kk, 293:443])

        # multiply by 2 for movement phase because there are half as many samples
        z1 = (pre_sum - 2 * move_sum) / pre_sum
        z2 = (post_sum - 2 * move_sum) / post_sum
        z3 = (post_sum - pre_sum) / post_sum

        new_rows.append({
            **base_row, 'atom_id': kk, 'u_hat': u, 'v_hat': v, 'dipole_gof': gof,
            'dipole_pos_x': pos[0], 'dipole_pos_y': pos[1], 'dipole_pos_z': pos[2],
            'dipole_ori_x': ori[0], 'dipole_ori_y': ori[1], 'dipole_ori_z': ori[2],
            'pre-move_change': z1, 'post-move_change': z2, 'post-pre_change': z3,
            'focal': (gof >= 95), 'rebound': (z3 >= 0.1),
            'movement_related': (z1 >= 0. and z2 >= 0.6)
        })

    return new_rows


SUBJECT_ID = [f.name.split('-')[1] for f in BIDS_ROOT.iterdir() if
              (not f.is_file()) and (f.name[:6] == 'sub-CC')]

new_rows = Parallel(n_jobs=N_JOBS, verbose=1)(
    delayed(procedure)(this_subject_id) for this_subject_id in SUBJECT_ID)

df_res = pd.DataFrame()
for new_row in new_rows:
    df_res = df_res.append(new_row, ignore_index=True)

# df_res.to_csv(RESULTS_DIR / 'atoms_cdl_res.csv')
pickle.dump(df_res, open(RESULTS_DIR / 'atoms_cdl_res.pkl', "wb"))
