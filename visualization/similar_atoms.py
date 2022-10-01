import numpy as np 
import pickle 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.signal as ss

subID = 'CC620490'

#Read in cdl model from pickle file 
fileName = '/media/NAS/lpower/CSC/results/' + subID + '/CSCraw_0.5s_20atoms.pkl'
cdl_model, info, allZ, z_hat_ = pickle.load(open(fileName,"rb"))

#Loop through atoms and find the unique set of atoms 
unique_atoms = [0] 
unique = True 
for atom in range(1, cdl_model.u_hat_.shape[0]): 
    print(atom) 
    for atom2 in unique_atoms: 
        print(atom2) 
        u_corrcoef = np.corrcoef(cdl_model.u_hat_[atom], cdl_model.u_hat_[atom2])[0,1] 
        v_corrcoef = np.max(ss.correlate(cdl_model.v_hat_[atom], cdl_model.v_hat_[atom2]))          

        if (abs(u_corrcoef) > 0.8) & (abs(v_corrcoef) > 0.8): 
            unique = False 
            print('same')  
    
    if unique == True:  
        unique_atoms.append(atom) 
    
    unique = True                           

#compare each atom to each other atom and find the correlation coefficient betw     een u vectors (spatial pattern)
coefficients = []
for atom in range(0, cdl_model.u_hat_.shape[0]):
    small_coefs = []
    for atom2 in range(0, cdl_model.u_hat_.shape[0]):
        corrcoef = np.corrcoef(cdl_model.u_hat_[atom], cdl_model.u_hat_[atom2])[0,1]
        small_coefs.append(abs(corrcoef))
    coefficients.append(small_coefs)
coefficients = np.asarray(coefficients)

#Plot data as a colourmap
viridis = cm.get_cmap('Blues', 12)
fig, ax = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)
plt.title('U correlations: Subject' + str(subID))
plt.xticks(np.arange(0,20))
plt.yticks(np.arange(0,20))
corrmap = ax.pcolormesh(coefficients, cmap=viridis, rasterized=True, vmin=0, vmax=1)
fig.colorbar(corrmap, ax=ax)
plt.show()

#Atoms are considered to have similar u vectors if the correlation coefficients      between atoms is greater than 0.8
#less than 0.99 eliminates the comparison of atoms with themselves
u_sim = np.where((coefficients > 0.8) & (coefficients < 1.))

#The same procedure of calculating correlation coefficients is repeated now with the psd of the v vector (frequency composition of the temporal pattern)
coefficients = []
for atom in range(0, cdl_model.v_hat_.shape[0]):
    small_coefs = []
    for atom2 in range(0, cdl_model.v_hat_.shape[0]):
        corrcoef = np.max(ss.correlate(cdl_model.v_hat_[atom],cdl_model.v_hat_[atom2]))
        small_coefs.append(abs(corrcoef))
    coefficients.append(small_coefs)
coefficients = np.asarray(coefficients)

#Plot data as a colourmap
viridis = cm.get_cmap('YlOrRd', 12)
fig, ax = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)
plt.title('PSD correlations: Subject' + str(subID))
plt.xticks(np.arange(0,20))
plt.yticks(np.arange(0,20))
corrmap = ax.pcolormesh(coefficients, cmap=viridis, rasterized=True, vmin=0, vmax=1)
fig.colorbar(corrmap, ax=ax)
plt.show()


v_sim = np.where((coefficients > 0.8) & (coefficients < 1.))

#restructure the lists of similar u and v vectors into a list of tuples (pairs      of similar atoms)
u_pairs = []
for atom in range(0, len(u_sim[0])):
    u_pair = (u_sim[0][atom], u_sim[1][atom])
    u_pairs.append(u_pair)

v_pairs = []
for atom in range(0, len(v_sim[0])):
    v_pair = (v_sim[0][atom], v_sim[1][atom])
    v_pairs.append(v_pair)

#Finds pairs that are considered similar in both their u and v vectors
same = list(set(v_pairs).intersection(u_pairs))
print(same)

