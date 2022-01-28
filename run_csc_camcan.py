# %%
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Memory, hash
import json
import os.path 

import matplotlib.pyplot as plt
import pickle
import mne
from mne_bids import BIDSPath, read_raw_bids
from alphacsc.viz.epoch import make_epochs

from utils_csc import run_csc
from utils_plot import plot_csc
#from utils_dripp import get_dripp_results

DATA_DIR = Path("/media/WDEasyStore/timb/camcan/release05/cc700/meg/pipeline/release005/")
BIDS_ROOT = DATA_DIR / "BIDSsep/smt/"  # Root path to raw BIDS files
SSS_CAL_FILE = Path("/home/timb/camcan/camcanMEGcalibrationFiles/sss_cal.dat")
CT_SPARSE_FILE = Path("/home/timb/camcan/camcanMEGcalibrationFiles/ct_sparse.fif")
PARTICIPANTS_FILE = BIDS_ROOT / "participants.tsv"
HOME_DIR = Path("/media/NAS/lpower/CSC/")

mem = Memory('.')

participants = pd.read_csv(PARTICIPANTS_FILE, sep='\t', header=0)

#Create pandas dataframe to hold the atom info that we will save to a csv
atomsDf = pd.DataFrame(columns=['Subject ID', 'Atom number', 'Dipole GOF', 'Dipole Pos x','Dipole Pos y', 'Dipole Pos z',
       'Dipole Ori x', 'Dipole Ori y', 'Dipole Ori z', 'Focal', 'Pre-Move Change', 'Post-Move Change', 'Post-Pre Change', 
        'Movement-related', 'Rebound', 'Unique'])
dataframe_dir = HOME_DIR / "results/atomData.csv"

atomsDf = pd.read_csv(dataframe_dir)

#Set up loop to run subset of camcan subjects
goodSubs = '/home/timb/camcan/proc_data/demographics_goodSubjects.csv'
goodSubs_df = pd.read_csv(goodSubs)
goodSubs_df = goodSubs_df.loc[goodSubs_df['DataReads']==1]
subjects = goodSubs_df['SubjectID'].tolist()
subjects = ['CC410286','CC420364']
go = True 

for subject_id in subjects:
    
    testPath = '/media/NAS/lpower/BetaSourceLocalization/preStimData/MEG0221/'+subject_id 
    currPath = '/media/NAS/lpower/CSC/results/' + subject_id + '/CSCraw_0.5s_20atoms.pkl'

    #Need to check that current subject is in participants list #
    test_id = 'sub-' + subject_id
    if test_id in participants['participant_id'].tolist():

        if go: #os.path.exists(testPath): 
            if go: #not os.path.exists(currPath):
        
                print(subject_id)
                plt.close()
        
                if len(sys.argv) > 1:  # get subject_id from command line
                    try:
                        subject_id_idx = int(sys.argv[1])
                        subject_id = participants.iloc[subject_id_idx]['participant_id']
                        subject_id = subject_id.split('-')[1]
                    except ValueError:
                        pass

                age, sex = participants[participants['participant_id'] == 'sub-' + str(subject_id)][['age', 'sex']].iloc[0]

                print(f'Running CSC pipeline on: {subject_id}, {str(age)} year old {sex}')
                fig_title = f'Subject {subject_id}, {str(age)} year old {sex}'

                # %% Parameters
                ch_type = "grad"  # run CSC
                sfreq = 150.
                atom_dur = 0.5
                n_atoms = 20

                # Epoching parameters
                tmin = -1.7
                tmax = 1.7
                baseline = (-1.25, -1.0)

                activation_tstart = -tmin
                shift_acti = True  # put activation to the peak amplitude time in the atom

                exp_params = {                 # in Tim's code:
                    "subject_id": subject_id,  # "CC620264"  - 76.33 y.o. female
                    "sfreq": sfreq,            # 300
                    "atom_duration": atom_dur,      # 0.5,
                    "n_atoms": n_atoms,             # 25,
                    "reg": 0.2,                # 0.2,
                    "eps": 1e-5,               # 1e-4,
                    "tol_z": 1e-3,             # 1e-2
                }   

                cdl_params = {
                    'n_atoms': exp_params['n_atoms'],
                    'n_times_atom': int(np.round(exp_params["atom_duration"] * sfreq)),
                    'rank1': True, 'uv_constraint': 'separate',
                    'window': True,  # in Tim's: False
                    'unbiased_z_hat': True,  # in Tim's: False
                    'D_init': 'chunk',
                    'lmbd_max': 'scaled', 'reg': exp_params['reg'],
                    'n_iter': 100, 'eps': exp_params['eps'],
                    'solver_z': 'lgcd',
                    'solver_z_kwargs': {'tol': exp_params['tol_z'], 'max_iter': 1000},
                    'solver_d': 'alternate_adaptive',
                    'solver_d_kwargs': {'max_iter': 300},
                    'sort_atoms': True,
                    'verbose': 1,
                    'random_state': 0,
                    'use_batch_cdl': True,
                    'n_splits': 10,
                    'n_jobs': 5
                }

                dripp_params = {
                    'threshold': 0,
                    'lower': 0, 'upper': 1,
                    'sfreq': sfreq,
                    'initializer': 'smart_start',
                    'alpha_pos': True,
                    'n_iter': 150,
                    'verbose': False,
                    'disable_tqdm': False
                }

                all_params = [exp_params, cdl_params, dripp_params]

                # Create folder to save results for the considered subject
                subject_output_dir = HOME_DIR / "results" / subject_id
                subject_output_dir.mkdir(parents=True, exist_ok=True)
                # Create folder to save final figures for a particular set of parameters
                exp_output_dir = subject_output_dir / hash(all_params)
                exp_output_dir.mkdir(parents=True, exist_ok=True)
                # Save experiment parameters
                with open(exp_output_dir / 'exp_params', 'w') as fp:
                    json.dump(all_params, fp, sort_keys=True, indent=4)

                # %% Read raw data from BIDS file
                bp = BIDSPath(
                    root=BIDS_ROOT,
                    subject=subject_id,
                    task="smt",
                    datatype="meg",
                    extension=".fif",
                    session="smt",
                )   
                raw = read_raw_bids(bp)

                # %% Preproc data

                raw.load_data()
                raw.filter(l_freq=None, h_freq=125)
                raw.notch_filter([50, 100])
                raw = mne.preprocessing.maxwell_filter(raw, calibration=SSS_CAL_FILE,
                                       cross_talk=CT_SPARSE_FILE,
                                       st_duration=10.0)

                # %% Now deal with Epochs

                all_events, all_event_id = mne.events_from_annotations(raw)
#           all_event_id = {'audiovis/1200Hz': 1,
#                 'audiovis/300Hz': 2,
#                 'audiovis/600Hz': 3,
#                 'button': 4,
#                 'catch/0': 5,
#                 'catch/1': 6}

                # for every button event,
                metadata_tmin, metadata_tmax = -3., 0
                row_events = ['button']
                keep_last = ['audiovis']

                metadata, events, event_id = mne.epochs.make_metadata(
                events=all_events, event_id=all_event_id,
                tmin=metadata_tmin, tmax=metadata_tmax, sfreq=raw.info['sfreq'],
                row_events=row_events, keep_last=keep_last)

                epochs = mne.Epochs(
                    raw, events, event_id, metadata=metadata,
                    tmin=tmin, tmax=tmax,
                    baseline=baseline,
                    preload=True, verbose=False
                )

                # "good" button events in Tim's:
                # button event is at most one sec. after an audiovis event,
                # and with at least 3 sec. between 2 button events.
                epochs = epochs["event_name == 'button' and audiovis > -1. and button == 0."]
                # epochs = epochs["event_name == 'button' and audiovis > -3."]

# XXX
#if True:
 #   evokeds = epochs.average()
  #  fig = evokeds.plot_joint(picks="grad")
   # fig_name = "evokeds.pdf"
    #fig.savefig(subject_output_dir / fig_name, dpi=300)

                # %% Setup CSC on Raw

                raw_csc = raw.copy()  # make a copy to run CSC on ch_type
                raw_csc.pick([ch_type, 'stim'])

                # Band-pass filter the data to a range of interest
                raw_csc.filter(l_freq=2, h_freq=45)
                raw_csc, events_csc = raw_csc.resample(
                sfreq, npad='auto', verbose=False, events=epochs.events)

                X = raw_csc.get_data(picks=['meg'])

                # %% Run multivariate CSC

                cdl_model, z_hat_ = mem.cache(run_csc)(X, **cdl_params)

                # %% Get and plot CSC results

                print("Get CSC results")

                # events here are only "good" button events
                events_no_first_samp = events_csc.copy()
                events_no_first_samp[:, 0] -= raw_csc.first_samp
                info = raw_csc.info.copy()
                info["events"] = events_no_first_samp
                info["event_id"] = None
                # atom_duration = exp_params.get("atom_duration", 0.7)
                allZ = make_epochs(
                    z_hat_,
                    info,
                    t_lim=(-activation_tstart, activation_tstart),
                    # n_times_atom=int(np.round(atom_duration * sfreq)),
                    n_times_atom=cdl_params['n_times_atom'])

                pkl_name = 'CSCraw_' + str(atom_dur) + 's_' + str(n_atoms) + 'atoms.pkl'
                outputFile = subject_output_dir / pkl_name
     
                info = raw.info
                pickle.dump([cdl_model, info, allZ, z_hat_], open(outputFile, "wb"))

        # %%
        #df_dripp = get_dripp_results(cdl_model,
#                             z_hat_,
#                             sfreq,
#                             events=events_no_first_samp,
#                             event_id=all_event_id['button'],
#                             dripp_params=dripp_params,
#                             save_dir=exp_output_dir)

# %%

                plot_csc(cdl_model=cdl_model,
                    raw_csc=raw_csc,
                    allZ=allZ,
                    shift_acti=shift_acti,
                    plot_acti_histo=True,
                    activation_tstart=activation_tstart,
                    #df_dripp=df_dripp,
                    save_dir=exp_output_dir,
                    title=fig_title, show=False)
                '''   
                fileName = '/media/NAS/lpower/CSC/results/' + subject_id + '/CSCraw_0.5s_20atoms.pkl'
                cdl_model, info, allZ, z_hat_ = pickle.load(open(fileName,"rb"))
                '''
                ##### Save important atom data into a csv file ####
                #Each atom is a row with columns: subjectID, atom #, correlation coefficient to each other atom (for u and PSD - 40), 
                #                                 unique boolean, dipole goodness of fit, dipole position, dipole orientation, 
                #                                 focal boolean, percent change between phases (3 cols), movement-related boolean,
                #                                rebound boolean
    
                #Thresholding variables 
                pre_move_thresh = 0.
                post_move_thresh = 0.6
                rebound_thresh = 0.1
                gof_thresh = 95
                n_atoms = 20

                #Files for dipole fitting
                subjectsDir = '/home/timb/camcan/subjects'
                transFif = subjectsDir + '/coreg/sub-' + subject_id + '-trans.fif'
                bemFif = subjectsDir + '/sub-' + subject_id + '/bem/sub-' + subject_id + '-5120-bem-sol.fif'
                fifFile = '/transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo.fif'
                epochFif = '/home/timb/camcan/proc_data/TaskSensorAnalysis_transdef/' + subject_id + fifFile
    
                #Fit a dipole for each atom 
                #Read in epochs for task data 
                epochs = mne.read_epochs(epochFif)
                epochs.pick_types(meg = 'grad')
                cov = mne.compute_covariance(epochs)
                info = epochs.info
     
                #Make an evoked object with all atom topographies for dipole fitting 
                evoked = mne.EvokedArray(cdl_model.u_hat_.T, info)
     
                #Fit dipoles
                dip = mne.fit_dipole(evoked, cov, bemFif, transFif,  verbose=False)[0]
                focalAtoms = np.where(dip.gof >= gof_thresh)[0]

                #loop through each atom and pull out important info about that atom
                for atom in range(0, n_atoms):
        
                    #Variables 
                    focal = False
                    u_coefs = []
                    psd_coefs = []
                    unique = True
                    rebound = False
                    movement_related = False 

                    #Set dipole characteristics
                    if atom in focalAtoms:
                        focal = True
                    gof = dip.gof[atom]
                    pos = dip.pos[atom] # 3-element list (index to x, y, z)
                    ori = dip.ori[atom] # 3-element list (index to x, y, z)

                    #Calculate the percent change in activation between different phases of movement
                    pre = allZ[:,atom,68:218] # -1.25 to -0.25 sec (150 samples)
                    move = allZ[:,atom,218:293] # -0.25 to 0.25 sec (75 samples)
                    post = allZ[:,atom,293:443] # 0.25 to 1.25 sec (150 samples)
        
                    pre_sum = np.sum(pre)
                    move_sum = np.sum(move)
                    post_sum = np.sum(post)
     
                    z1 = (pre_sum-move_sum*2)/pre_sum #multiply by 2 for movement phase because there are half as many samples 
                    z2 = (post_sum-move_sum*2)/post_sum
                    z3 = (post_sum-pre_sum)/post_sum
    
                    #Check if movement and rebound conditions are met
                    if (z1 >= pre_move_thresh and z2 >= post_move_thresh):
                        movement_related = True

                        if z3 >= rebound_thresh:
                            rebound = True

                    #Calculate each u_vector and psd coefficient for the relationship to each other atom
                    for atom2 in range(0, cdl_model.u_hat_.shape[0]):
                        #calculate and save u coefficients to list
                        corrcoef = np.corrcoef(cdl_model.u_hat_[atom], cdl_model.u_hat_[atom2])[0,1]
                        u_coefs.append(abs(corrcoef))
            
                        #calculate and save psd coefficients to list
                        psd1 = np.abs(np.fft.rfft(cdl_model.v_hat_[atom], n=256)) ** 2
                        psd2 = np.abs(np.fft.rfft(cdl_model.v_hat_[atom2], n=256)) ** 2
                        corrcoef = np.corrcoef(psd1, psd2)[0,1]
                        psd_coefs.append(abs(corrcoef))

                    #Checks if atom is similar to any other atoms in the list 
                    #(has coefficient greater than 0.8 for both u vector and psd)
                    u_coefs = np.asarray(u_coefs)
                    psd_coefs = np.asarray(psd_coefs)
                    u_sim = np.where((u_coefs > 0.8) & (u_coefs < 1.))[0]
                    v_sim = np.where((psd_coefs > 0.8) & (psd_coefs < 1.))[0]
                    same = list(set(u_sim).intersection(v_sim))
                    #If any similar atoms are found, set unique boolean to false
                    if len(same)>0:
                        unique = False 

                    #Create dictionary with atom information and append to dataframe
                    atom_dict = {'Subject ID': subject_id, 'Atom number': atom, 'Dipole GOF': gof, 'Dipole Pos x': pos[0],'Dipole Pos y': pos[1], 'Dipole Pos z': pos[2],
                    'Dipole Ori x': ori[0], 'Dipole Ori y': ori[1], 'Dipole Ori z': ori[2], 'Focal': focal, 'Pre-Move Change': z1, 'Post-Move Change': z2, 
                    'Post-Pre Change': z3,'Movement-related': movement_related, 'Rebound': rebound, 'u_coef 0': u_coefs[0], 'u_coef 1': u_coefs[1],'u_coef 2': u_coefs[2],
                    'u_coef 3': u_coefs[3],'u_coef 4':u_coefs[4], 'u_coef 5':u_coefs[5],'u_coef 6': u_coefs[6], 'u_coef 7': u_coefs[7], 'u_coef 8': u_coefs[8], 
                    'u_coef 9': u_coefs[9], 'u_coef 10': u_coefs[10], 'u_coef 11': u_coefs[11],'u_coef 12': u_coefs[12], 'u_coef 13': u_coefs[13],'u_coef 14': u_coefs[14],
                    'u_coef 15': u_coefs[15], 'u_coef 16': u_coefs[16], 'u_coef 17': u_coefs[17], 'u_coef 18': u_coefs[18],'u_coef 19': u_coefs[19], 'psd_coef 0':psd_coefs[0],
                    'psd_coef 1': psd_coefs[1],'psd_coef 2': psd_coefs[2],'psd_coef 3': psd_coefs[3],'psd_coef 4': psd_coefs[4], 'psd_coef 5': psd_coefs[5], 
                    'psd_coef 6': psd_coefs[6], 'psd_coef 7': psd_coefs[7], 'psd_coef 8': psd_coefs[8],'psd_coef 9': psd_coefs[9], 'psd_coef 10': psd_coefs[10], 
                    'psd_coef 11': psd_coefs[11],'psd_coef 12': psd_coefs[12], 'psd_coef 13': psd_coefs[13], 'psd_coef 14': psd_coefs[14],'psd_coef 15': psd_coefs[15], 
                    'psd_coef 16': psd_coefs[16], 'psd_coef 17': psd_coefs[17], 'psd_coef 18': psd_coefs[18],'psd_coef 19': psd_coefs[19],'Unique': unique}
                    atomsDf = atomsDf.append(atom_dict, ignore_index=True)

                #Save dataframe as a csv file 
                atomsDf.to_csv(dataframe_dir)
   
    '''
    ##### #Recreate MEG data for each atom ##### 
    atom_data = []
     
    #Reconstructs MEG data by convolving the v vector (short temporal pattern) with the z (activation) vector and then multiplying by the u vector 
    #Yields data that is number of channels x number of samples 
    for atom in range(0,n_atoms):
        u = cdl_model.u_hat_[atom]
        v = cdl_model.v_hat_[atom]
        z = z_hat_[:,atom,:]
        z = z.flatten() #z is trials x time so flatten to give entire timecourse 
     
        conv = np.convolve(v, z, mode='valid')
        dat = np.multiply.outer(u, conv) #an array of channels x time 
     
        #Create a raw object from the atom data 
        raw_csc.pick_types(meg='grad')
        raw_atom = mne.io.RawArray(dat, raw_csc.info)
    
        epochs = mne.Epochs(raw_atom, events, event_id, metadata=metadata, tmin=tmin, tmax=tmax, baseline=baseline, preload=True, verbose=False)

        #"good" button event - button event is at most one sec after an audiovis event and with at least 3 sec between 2 button events
        epochs = epochs["event_name == 'button' and audiovis > -1. and button == 0."]

        #Create TFR
        fmin = 2
        fmax = 45
        freqs = np.arange(fmin,fmax)
        n_cycles = freqs/2.
        power, itc = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=True, decim=3)
    
        #save for plotting later
        atom_data.append(power)
    '''
