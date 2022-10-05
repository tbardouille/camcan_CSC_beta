# %%
from alphacsc.utils.convolution import construct_X, construct_X_multi
from alphacsc.utils.dictionary import get_D
from alphacsc import BatchCDL, GreedyCDL
import alphacsc
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

from sklearn import cluster
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt

from joblib import Memory, Parallel, delayed

import mne
from utils_csc import get_atom_df, correlation_clustering_atoms, get_df_mean, reconstruct_class_signal, get_cdl_pickle_name
from utils_plot import plot_mean_atom
from config import CDL_PARAMS, RESULTS_DIR, PARTICIPANTS_FILE

import os

# read clustering results
OUTPUT_DIR = '/media/NAS/lpower/CSC/results/u_'
threshold = u_thresh = v_thresh = 0.4
sfreq=150.0

csv_dir = OUTPUT_DIR + \
    str(u_thresh) + '_v_' + str(v_thresh) + '_groupSummary.csv'
if not os.path.exists(csv_dir):
    print(f'{csv_dir} does not exist')
    # atom_df = get_atom_df(RESULTS_DIR, PARTICIPANTS_FILE)
    # groupSummary, atomGroups = correlation_clustering_atoms(
    #     atom_df, threshold=threshold, output_dir=OUTPUT_DIR)

groupSummary = pd.read_csv(csv_dir)

# select big enough groups
#threshold_group = .2
#total_subjects = groupSummary['Number of Atoms'].sum() / 20
#group_id = groupSummary[groupSummary['Number of Subjects']
#                        > threshold_group * total_subjects]['Group Number'].values

#group_id = [  2,   6,   9,  10,  12,  13,  14,  20,  21,  23,  24,  26,  29,
#        31,  32,  33,  34,  36,  37,  38,  39,  48,  53,  57,  58,  60,
#        61,  63,  64,  70,  71,  72,  80,  82,  83,  87,  88,  89,  94,
#        98, 101, 112, 118, 121, 125, 126, 129, 136, 140, 143, 146, 147,
#       148, 149, 150, 151, 152, 154, 155, 157, 161, 162, 163, 164, 170,
#       172, 179, 181, 197, 203, 204, 205, 208, 215, 216, 217, 218, 219,
#       224]
#group_id = [2,9,13,23,61,70,126,148,151,154,205,208,216]
#group_id = [12,21,29,31,34]
group_ids = [13]
for group_id in group_ids:
    #print(total_subjects)
    # Save atomGroups to dataframe
    csv_dir = OUTPUT_DIR + \
        str(u_thresh) + '_v_' + str(v_thresh) + '_atomGroups.csv'
    atomGroups = pd.read_csv(csv_dir)

    atomGroups.rename(columns={'Subject ID': 'subject_id'}, inplace=True)
    atomGroups.rename(columns={'Atom number': 'atom_id'}, inplace=True)
    clustering_df = atomGroups[atomGroups['Group number']==group_id]

    #Reconstruct the MEG timecourse for this set of data
    X, n_times_atom = reconstruct_class_signal(df = clustering_df, results_dir=RESULTS_DIR) 

    #Concatenate V and Z vectors across all relevant atoms 
    df = clustering_df
    results_dir = RESULTS_DIR
    Z_temp = []
    V_temp = []
    min_n_times_valid = np.inf
    for subject_id in set(df['subject_id'].values):
        print(subject_id)
        file_name = results_dir / subject_id / get_cdl_pickle_name()
        cdl_model, info, allZ, _ = pickle.load(open(file_name, "rb"))
        atom_idx = df[df['subject_id'] == subject_id]['atom_id'].values.astype(int)
        print(atom_idx)
        Z_temp.append(allZ[:,atom_idx,:])
        min_n_times_valid = min(min_n_times_valid, Z_temp[-1].shape[0])
        V_temp.append(cdl_model.v_hat_[atom_idx, :])
        
    # combine z and v vectors
    Z = Z_temp[0][:min_n_times_valid, :, :]
    V = V_temp[0]
    for this_z, this_v in zip(Z_temp[1:], V_temp[1:]):
        this_z = this_z[:min_n_times_valid,:,:]
        Z = np.concatenate((Z, this_z), axis=1)
        V = np.concatenate((V, this_v), axis=0)
    n_times_atom = V.shape[-1]

    #Convolve the v and z vectors to create concatenated timecourse 
    #Do this in a loop for each epoch to maintain epochs shape for TFR 
    dat_list = []
    for x in range(0,Z.shape[0]):
        print(x)
        for x2 in range(0,Z.shape[1]):
            print(x2)
            Z_temp = Z[x,x2,:]
            Z_temp = Z_temp.flatten()
            V_temp = V[x2,:]
            V_temp = V_temp.flatten()
            dat = np.convolve(V_temp, Z_temp, mode='same')
            dat_list.append(dat)

    #Reshape epochs array from convolved data
    dat_arr = np.asarray(dat_list)
    dat_arr = np.reshape(dat_arr, (dat_arr.shape[0], 1, dat_arr.shape[1]))

    #Calculate TFR on reconstructed data
    freqs = np.arange(2,45,0.5)  
    n_cycles = freqs/2
    power = mne.time_frequency.tfr_array_morlet(dat_arr, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles,output='avg_power')

    #Cut off edges to avoid edge effects
    power = power[:,:,30:481] #remove 200 ms from start and end (-1.5 to 1.5 seconds)

    #Apply baseline to TFR
    times = np.arange(0,power.shape[2])/sfreq-1.5 
    baseline_power = mne.baseline.rescale(power, times, baseline=(-1.5, -0.5), mode='logratio') 

    '''
    #Plot TFR
    fig, ax = plt.subplots()
    x, y = mne.viz.centers_to_edges(times, freqs)
    mesh = ax.pcolormesh(x, y, (baseline_power[0]), cmap='RdBu_r',vmin=-0.5, vmax=0.5)#, vmin=vmin, vmax=vmax)
    ax.set_title('TFR concatenated MEG data')
    ax.set(ylim=(5,25), xlabel='Time (s)')
    fig.colorbar(mesh)
    plt.tight_layout()

    plt.show()
    plt.savefig('/media/NAS/lpower/CSC/camcan_CSC_beta-pipeline_mean/tfr_plot_' + str(group_id)+ 'v2.png')
    '''
    #Plot TFR
    fig, ax = plt.subplots()
    x, y = mne.viz.centers_to_edges(times, freqs)
    mesh = ax.pcolormesh(x, y, 10*np.log10(power[0]), cmap='jet',vmin=-240, vmax=-210)#, vmin=vmin, vmax=vmax)
    ax.set_title('TFR concatenated MEG data')
    ax.set(ylim=freqs[[0, -1]], xlabel='Time (s)')
    fig.colorbar(mesh)
    plt.tight_layout()

    plt.show()
    plt.savefig('/media/NAS/lpower/CSC/camcan_CSC_beta-pipeline_mean/tfr_plot_' + str(group_id)+ '.png')
