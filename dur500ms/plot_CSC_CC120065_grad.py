import pickle
import matplotlib.pyplot as plt
import numpy as np
import mne
import os
from scipy.signal import tukey
import pandas as pd
import multiprocessing as mp
import scipy.stats as ss   

inputDir = '/home/timb/data/camcan/proc_data/TaskSensorAnalysis_transdef'
outputDir =  '/home/timb/data/CSC'
n_atoms = 25
subjectID = 'CC120065'
sensorType = 'grad'

preStimStart = 0  # -1.7 seconds wrt cue
activeStart = 255 # 0 seconds wrt cue
zWindowDuration = 75	# 0.5 seconds

subjectOutputDir = os.path.join(outputDir, subjectID)

# Load in the CSC results
megFile = os.path.join(inputDir, subjectID, 'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo.fif')
outputFile = os.path.join(subjectOutputDir, 'CSCepochs_' 
        +sensorType  + str(n_atoms) + 'atoms.pkl')
cdl, info = pickle.load(open(outputFile, "rb"))

sfreq = info['sfreq']


# Plot time course for all atoms
fig, axes = plt.subplots(5, 5, sharex=True, sharey=True)
for i_atom in range(n_atoms):
    ax = np.ravel(axes)[i_atom]
    v_hat = cdl.v_hat_[i_atom]
    t = np.arange(v_hat.size) / sfreq
    ax.plot(t, v_hat)
    ax.set(title='Atom #' + str(i_atom))
    ax.grid(True)
plt.show()

# Plotting sensor weights from U vectors as topographies
fig, axes = plt.subplots(5, 5, figsize=(11,8.5))
for i_atom in range(n_atoms):
    ax = np.ravel(axes)[i_atom]
    u_hat = cdl.u_hat_[i_atom]
    mne.viz.plot_topomap(u_hat, info, axes=ax, show=False)
    ax.set(title='Atom #' + str(i_atom))
plt.show()


# Plot z vector overlaid for all trials (for each atom)
# first, make a time vector for the x-axis
t1 = np.arange(cdl.z_hat_.shape[2])/sfreq-1.7
# Then, plot each atom's Z
fig, axes = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(11,8.5))
for i_atom in range(n_atoms):
    ax = np.ravel(axes)[i_atom]
    z_hat = cdl.z_hat_[:,i_atom,:]
    ax.plot(t1, z_hat.T)
    ax.set(title='Atom #' + str(i_atom))
plt.show()

# Calculate the sum of all Z values for prestim and active intervals
allZ = cdl.z_hat_         
activezHat = allZ[:,:,activeStart:activeStart+zWindowDuration] 
preStimzHat = allZ[:,:,preStimStart:preStimStart+zWindowDuration] 
activeSum =np.sum(np.sum(activezHat,axis=2),0)     
preStimSum = np.sum(np.sum(preStimzHat,axis=2),0)   
diffZ = activeSum-preStimSum   
plt.figure()
plt.bar(x=range(25), height=diffZ) 
plt.show()   


# T-test of z-value sums across trials between intervals
activeSumPerTrial = np.sum(activezHat,axis=2)
preStimSumPerTrial = np.sum(preStimzHat,axis=2)
diffZperTrial = activeSumPerTrial-preStimSumPerTrial
tstat, pvalue = ss.ttest_1samp(diffZperTrial, popmean=0, axis=0) 
plt.figure()
plt.bar(x=range(25), height=tstat) 
plt.grid(True)
plt.show()

