import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import scipy.signal as ss
import os

#subjects = ['CC110037','CC110606','CC120166','CC120313','CC120727','CC121428','CC120640','CC210657','CC220518','CC220999',
#            'CC220107','CC222125','CC223286','CC310256','CC320089','CC320429','CC320621','CC321000','CC321431','CC321899',
#            'CC410287','CC420060','CC420157','CC420217','CC420383','CC420566','CC420729','CC510304','CC510438','CC520083',
#            'CC520215','CC520503','CC520597','CC520980','CC610061','CC610288','CC610576','CC620114','CC620264','CC620885',
#            'CC621642','CC710088','CC710342','CC710664','CC720290','CC721052','CC721504','CC722891','CC723395','CC221107']

subDir = '/media/NAS/lpower/CSC/results'
subjects = next(os.walk(subDir))[1]

#Set parameters
u_thresh = 0.8
v_thresh = 0.8

numGroups_list = []
exclude_list = [] 

for subID in subjects: 
    print(subID)
    
    #Read in cdl model from pickle file
    fileName = '/media/NAS/lpower/CSC/results/' + subID + '/CSCraw_0.5s_20atoms.pkl'
    
    if os.path.exists(fileName):
        cdl_model, info, allZ, z_hat_ = pickle.load(open(fileName,"rb"))
        
        atomGroups = pd.DataFrame(columns=['Subject ID', 'Atom number', 'Group number'])

        atomNum = 0
        groupNum = 0
        unique = True

        #Create a row in the dataframe for the first atom, placing it in group 0
        groupDict = {'Subject ID': subID, 'Atom number': atomNum, 'Group number': groupNum}
        atomGroups = atomGroups.append(groupDict, ignore_index=True)

        #For each atom, checks if it is correlated to any atoms that have already been sorted and sorts accordingly
        for atom in range(1, cdl_model.u_hat_.shape[0]):
            atomNum = atom  
        
            max_corr = 0
            max_group = 0 

            #Loops through groups that have already been created and checks current atom's average  correlation
            for group in range(0, groupNum+1):
                gr_atoms = atomGroups[atomGroups['Group number'] == group]['Atom number'].tolist()
                u_coefs = []
                v_coefs = []
        
                #for each atom in the group, calculate the correlation coefficients and average them
                for atom2 in gr_atoms:
                    u_corrcoef = np.corrcoef(cdl_model.u_hat_[atom], cdl_model.u_hat_[atom2])[0,1]
                    u_coefs.append(u_corrcoef)

                    v_corrcoef = np.max(ss.correlate(cdl_model.v_hat_[atom],cdl_model.v_hat_[atom2]))
                    v_coefs.append(v_corrcoef)

                #average across u and psd correlation coefficients 
                u_coefs = abs(np.asarray(u_coefs))
                avg_u = np.mean(u_coefs)

                v_coefs = abs(np.asarray(v_coefs)) 
                avg_v = np.mean(v_coefs)

                #If U vector and PSD correlations are both high, sorts atom into that group
                if (avg_u > u_thresh) & (avg_v > v_thresh):
                    unique = False 
                    if (avg_u + avg_v) > max_corr:
                        max_corr = (avg_u + avg_v)
                        max_group = group
                
            #If a similar atom is not found, creates a new group and assigns the current atom to that group
            if (unique == False):
                groupDict = {'Subject ID': subID, 'Atom number': atomNum, 'Group number': max_group}
        
            if (unique == True): 
                groupNum = groupNum + 1
                groupDict = {'Subject ID': subID, 'Atom number': atomNum, 'Group number': groupNum}
        
            unique = True

            #Append data for current atom to dataframe 
            atomGroups = atomGroups.append(groupDict, ignore_index=True)

        #Summary statistics for the current dataframe:

        #Number of distinct groups 
        groups = atomGroups['Group number'].tolist()
        groups = np.asarray(groups)
        numGroups = len(np.unique(groups))
        print(numGroups)
        numGroups_list.append(numGroups)

        #Number of atoms per group
        numAtoms_list = []

        for un in np.unique(groups):
            numAtoms = len(np.where(groups==un)[0])
            numAtoms_list.append(numAtoms)

        numAtoms_list = np.asarray(numAtoms_list)
        meanAtoms = np.mean(numAtoms_list)
        stdAtoms = np.std(numAtoms_list)

        #print("Average number of atoms per group:")
        #print(str(meanAtoms) + " +/- " + str(stdAtoms))
        
        if numGroups<14:
            exclude_list.append(subID)

        groupSummary = pd.DataFrame(columns=['Group Number', 'Number of Atoms'])
        groupSummary['Group Number'] = np.unique(groups)
        groupSummary['Number of Atoms'] = numAtoms_list

        #Save group summary dataframe
        #outputDir = '/media/NAS/lpower/CSC/results/' + subID + '/u_' + str(u_thresh) + '_psd_' + str(psd_thresh) + '_groupSummary.csv'
        #groupSummary.to_csv(outputDir)

        #Save atomGroups to dataframe
        #outputDir = '/media/NAS/lpower/CSC/results/' + subID + '/u_' +  str(u_thresh) + '_psd_' + str(psd_thresh) + '_atomGroups.csv'
        #atomGroups.to_csv(outputDir)

print(exclude_list)

#Plot distribution of number of groups
bins = np.arange(0,21)
plt.hist(numGroups_list, bins=bins)
plt.xlabel("Number of Groups")
plt.ylabel("Frequency")
plt.show()

