import mne
import pandas as pd 
import pickle
import os
import numpy as np

#Files for loading data 
home_dir = '/media/NAS/lpower/CSC/'
results_dir = home_dir + 'results/'
atom_df_file = results_dir + 'df_mean_atom.csv'
subID = next(os.walk(results_dir))[1][0]
sample_data = results_dir + subID + '/CSCraw_0.5s_20atoms.pkl'

#Files for dipole fitting
subjectsDir = '/home/timb/camcan/subjects'
transFif = home_dir + 'fsaverage-trans.fif' #I downloaded this from MNE github - is this valid? 
bemFif = subjectsDir + '/fsaverage' + '/bem/fsaverage'+ '-5120-bem-sol.fif'
emptyroomFif = '/media/NAS/lpower/BetaSourceLocalization/emptyroomData/' + subID + '/emptyroom_trans-epo.fif' #taking empty room noise data from a random subject to compute the covariance (I don't know if this is valid)

#Read in necessary data
atom_df = pd.read_csv(atom_df_file)
_,info,_,_ = pickle.load(open(sample_data, "rb"))
meg_indices = mne.pick_types(info, meg='grad')
info = mne.pick_info(info, meg_indices)

#Compute noise covariance
empty_room = mne.read_epochs(emptyroomFif)
noise_cov = mne.compute_covariance(empty_room, tmin=0, tmax=None)

#Make sphere head model because we don't have coregistration info for the mean 
sphere = mne.make_sphere_model(r0='auto', head_radius='auto',info=info)

#Pull out the u vector data from the dataframe for each mean atom and reformat it (has to be significantly reformatted because it saved as a string in the dataframe)
for label in atom_df['label'].tolist(): 
    print(label)
    u_hat_string = atom_df[atom_df['label']==label]['u_hat'].values[0]
    u_hat_list = u_hat_string.split()
    if u_hat_list[0] == "[":
        u_hat_list = u_hat_list[1:]
    num_list = [] 
    for x in u_hat_list: 
        x = x.replace("[","")
        x = x.replace("]","")
        x = float(x)
        num_list.append(x)

    u_hat_array = np.asarray(num_list)
    u_hat_array = u_hat_array.reshape(len(u_hat_array),1)  
    
    #Create evoked object from u vector data 
    evoked = mne.EvokedArray(u_hat_array, info)
     
    #Fit dipoles
    dip = mne.fit_dipole(evoked, noise_cov, bemFif, transFif,  verbose=False)[0]
    dip.plot_locations(transFif,'fsaverage',subjectsDir)

