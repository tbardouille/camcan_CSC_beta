## Running the CSC analysis from the "mu" example on the alphaCSC website
#       This may not extract beta bursts because the interval is too long
#       but it should find mu bursts

import mne
import pickle
import os
from scipy.signal import tukey
import numpy as np

###############################################################################
# Important variables
# Define the shape of the dictionary
n_atoms = 25                        # Number of atoms
atomDuration = 0.5 # [seconds]      # Atom duration
sfreq = 300.                        # Resample MEG data to this rate for CSC analysis
sensorType = 'grad'                 # Use this sensor type for CSC analysis
subjectID = 'CC620264'              # Subject to analyse
 
# Paths
inputDir = '/home/timb/data/camcan/proc_data/TaskSensorAnalysis_transdef'
outputDir =  '/home/timb/data/CSC'


###############################################################################
# Read in an MEG dataset with ~60 trials
print('Reading MEG Data')
megFile = os.path.join(inputDir, subjectID, 'transdef_transrest_mf2pt2_task_raw_buttonPress_duration=3.4s_cleaned-epo.fif')

subjectOutputDir = os.path.join(outputDir, subjectID)
if not os.path.exists(subjectOutputDir):
	os.mkdir(subjectOutputDir)

outputFile = os.path.join(subjectOutputDir, 'CSCepochs_' + str(int(atomDuration*1000)) + 'ms_'  
	+ sensorType  + str(n_atoms) + 'atoms.pkl')


# Read in the data
epochs = mne.read_epochs(megFile)
epochs.pick_types(meg=sensorType)

# Band-pass filter the data to a range of interest
epochsBandPass = epochs.filter(l_freq =2 , h_freq=45)

# SKIP: notch filter to remove 50 Hz power line noise since LP filter is at 45 Hz
#epochsNotch = epochsBandPass.filter(l_freq=48, h_freq=52)
epochsNotch = epochsBandPass

# Downsample data to 150Hz to match alphacsc example
epochsResample = epochsNotch.resample(sfreq=sfreq)

n_times_atom = int(np.round(atomDuration * sfreq))

# Scale data prior to fitting with CSC
Y = epochsResample.get_data()
num_trials, num_chans, num_samples = Y.shape
Y *= tukey(num_samples, alpha=0.1)[None, None, :]
Y /= np.std(Y)

#epochs.plot()

###############################################################################
# Next, we define the parameters for multivariate CSC

print('Building CSC')
from alphacsc import BatchCDL
cdlMEG = BatchCDL(
    # Shape of the dictionary
    n_atoms=n_atoms,
    n_times_atom=n_times_atom,
    # Request a rank1 dictionary with unit norm temporal and spatial maps
    rank1=True, uv_constraint='separate',
    # Initialize the dictionary with random chunk from the data
    D_init='chunk',
    # rescale the regularization parameter to be 20% of lambda_max
    lmbd_max="scaled", reg=.2,
    # Number of iteration for the alternate minimization and cvg threshold
    n_iter=100, eps=1e-4,
    # solver for the z-step
    solver_z="lgcd", solver_z_kwargs={'tol': 1e-2, 'max_iter': 1000},
    # solver for the d-step
    solver_d='alternate_adaptive', solver_d_kwargs={'max_iter': 300},
    # Technical parameters
    verbose=1, random_state=0, n_jobs=16)

###############################################################################
# Fit the model and learn rank1 atoms
print('Running CSC')
cdlMEG.fit(Y)

info = epochsResample.info

# Save results of CSC with dataset info
pickle.dump([cdlMEG, info], open(outputFile, "wb"))

