import pickle
import matplotlib.pyplot as plt
import numpy as np
import mne
import os
from scipy.signal import tukey
import pandas as pd
import multiprocessing as mp
import scipy.stats as ss   

def makeCSCfigure(subjectID, cdl, tstat, diffZperTrial, tIndex, sfreq):

    # Setup a new figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(8.5,11))
    fig.suptitle(subjectID + ': Atom # ' + str(tIndex))

    # Plot u-vector as a topography in top-left corner
    u_hat = cdl.u_hat_[tIndex]
    mne.viz.plot_topomap(u_hat, info, axes=axes[0,0], show=False)
    axes[0,0].set_title('U vector')

    # Plot v-vector in top-right corner
    v_hat = cdl.v_hat_[tIndex]
    t = np.arange(v_hat.size) / sfreq
    axes[0,1].plot(t, v_hat)
    axes[0,1].set_title('V vector')
    axes[0,1].set_xlabel('Time [s]')
    axes[0,1].set_ylabel('v [a.u.]')
    axes[0,1].grid(True)

    # Plot histogram of Z difference in bottom left corner
    thisTStat = tstat[tIndex]
    axes[1,0].hist(diffZperTrial[:,tIndex])
    axes[1,0].set_title("Distribution of Z differences, t = {:.2f}".format(thisTStat))
    axes[1,0].set_ylabel('# occurrences')
    axes[1,0].set_xlabel('z [a.u.]')
    axes[1,0].grid(True)

    # Plot z-vector with one colour per trial in bottom right corner
    t1 = np.arange(cdl.z_hat_.shape[2])/sfreq-1.7
    z_hat = cdl.z_hat_[:,tIndex,:]
    axes[1,1].plot(t1, z_hat.T)
    axes[1,1].set_title('Z vector - 1 colour per trial')
    axes[1,1].set_xlabel('Time [s]')
    axes[1,1].set_ylabel('z [a.u.]')
    axes[1,1].grid(True)

    plt.show()

inputDir = '/home/timb/data/camcan/proc_data/TaskSensorAnalysis_transdef/'
outputDir =  '/home/timb/data/CSC'
n_atoms = 25
subjectID = 'CC120065'
sensorType = 'grad'

preStimStart = 0  # -1.7 seconds wrt cue
activeStart = 255 # 0 seconds wrt cue
zWindowDuration = 75	# 0.5 seconds

tThresh = -2

subjectOutputDir = os.path.join(outputDir, subjectID)

# Load in the CSC results
megFile = os.path.join(inputDir, subjectID, 'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo.fif')
outputFile = os.path.join(subjectOutputDir, 'CSCepochs_' 
        +sensorType  + str(n_atoms) + 'atoms.pkl')

cdl, info = pickle.load(open(outputFile, "rb"))

sfreq = info['sfreq']

# Calculate the sum of all Z values for prestim and active intervals
allZ = cdl.z_hat_         
activezHat = allZ[:,:,activeStart:activeStart+zWindowDuration] 
preStimzHat = allZ[:,:,preStimStart:preStimStart+zWindowDuration] 
activeSum =np.sum(np.sum(activezHat,axis=2),0)     
preStimSum = np.sum(np.sum(preStimzHat,axis=2),0)   
diffZ = activeSum-preStimSum   

# T-test of z-value sums across trials between intervals
activeSumPerTrial = np.sum(activezHat,axis=2)
preStimSumPerTrial = np.sum(preStimzHat,axis=2)
diffZperTrial = activeSumPerTrial-preStimSumPerTrial
tstat, pvalue = ss.ttest_1samp(diffZperTrial, popmean=0, axis=0) 

# Find atoms with t below the threshold
tIndices = np.where(tstat < tThresh)[0]

print('There are ' + str(len(tIndices)) + ' significant atoms.')

for tIndex in tIndices:
    makeCSCfigure(subjectID, cdl, tstat, diffZperTrial, tIndex, sfreq)


