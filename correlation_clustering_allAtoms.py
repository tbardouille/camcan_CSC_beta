import pickle
import numpy as np
import pandas as pd

#Set thresholds for current run
thresholds = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]


subjects = ['CC110606','CC120166','CC120313','CC120727','CC120640','CC210657','CC220518','CC220999','CC221107',
            'CC220107','CC222125','CC223286','CC310256','CC320089','CC320429','CC320621','CC321000','CC321431','CC321899',
            'CC410287','CC420060','CC420157','CC420217','CC420566','CC420729','CC510304','CC510438','CC520083',
            'CC520215','CC520503','CC520597','CC520980','CC610061','CC610288','CC610576','CC620114','CC620264','CC620885',
            'CC710088','CC710342','CC710664','CC720290','CC721052','CC721504','CC722891','CC723395']

u_vector_list = []
psd_list = []
count = 0

#Create a dataframe to hold subject, atom, and index codings 
atomData = pd.DataFrame(columns=['Subject ID', 'Atom number', 'Index'])

for subID in subjects: 
    
    #Read in cdl model for current subject
    fileName = '/media/NAS/lpower/CSC/results/' + subID + '/CSCraw_0.5s_20atoms.pkl'
    cdl_model, info, allZ, z_hat_ = pickle.load(open(fileName,"rb"))

    #for each atom, calculate the u vector and psd and add them to the global lists
    for atom in range(0, cdl_model.u_hat_.shape[0]):

        u_hat = cdl_model.u_hat_[atom]
        u_vector_list.append(u_hat)

        psd = np.abs(np.fft.rfft(cdl_model.v_hat_[atom], n=256)) ** 2
        psd_list.append(psd)
    
        #Update the dataframe and increment count 
        dfDict = {'Subject ID': subID, 'Atom number': atom, 'Index': count}
        atomData = atomData.append(dfDict, ignore_index = True)
        count = count+1

#Calculate the correlation coefficient between all atoms 
u_vector_list = np.asarray(u_vector_list)
psd_list = np.asarray(psd_list)

u_coefs = np.corrcoef(u_vector_list,u_vector_list)[0:920][0:920]
psd_coefs = np.corrcoef(psd_list, psd_list)[0:920][0:920]

for thresh in thresholds: 

    #Set parameters
    u_thresh = thresh
    psd_thresh = thresh

    atomNum = 0
    groupNum = 0
    unique = True

    #Make atom groups array to keep track of the group that each atom belongs to 
    atomGroups = pd.DataFrame(columns=['Subject ID','Atom number','Index','Group number'])

    #loop through each atom and find which group it belongs in 
    for ind in atomData['Index'].tolist():
        row = atomData[atomData['Index']==ind]
        print(row)

        subID = row['Subject ID'].tolist()[0]
        atomNum = row['Atom number'].tolist()[0]
     
        max_corr = 0
        max_group = 0
     
        #Loops through the existing groups and calculates the atom's average correlation to that group
        for group in range(0, groupNum+1):
            gr_atoms = atomGroups[atomGroups['Group number'] == group]
            inds = gr_atoms['Index'].tolist()
            u_groups = []
            psd_groups = []
     
            #Find the u vector and correlation coefficient comparing the current atom to each atom in the group
            for ind2 in inds:
                u_coef = u_coefs[ind][ind2]
                u_groups.append(u_coef)
     
                psd_coef = psd_coefs[ind][ind2]
                psd_groups.append(psd_coef)
     
            #average across u and psd correlation coefficients in that group
            u_groups = abs(np.asarray(u_groups))
            avg_u = np.mean(u_groups)
     
            psd_groups = abs(np.asarray(psd_groups))
            avg_psd = np.mean(psd_groups)

            #check if this group passes the thresholds 
            if (avg_u > u_thresh) & (avg_psd > psd_thresh):
                unique = False
                #If it does, also check if this is the highest cumulative correlation so far
                if (avg_u + avg_psd) > max_corr:
                    max_corr = (avg_u + avg_psd)
                    max_group = group
     
        #If the atom was similar to at least one group, sorts it into the group that it had the highest cumulative correlation to 
        if (unique == False):
            groupDict = {'Subject ID': subID, 'Atom number': atomNum, 'Index': ind, 'Group number': max_group}
     
        #If a similar group is not found, a new group is create and the current atom is added to that group
        elif (unique == True):
            groupNum = groupNum+1
            print(groupNum)
            groupDict = {'Subject ID': subID, 'Atom number': atomNum, 'Index': ind, 'Group number': groupNum}
     
        #Add to group dataframe and reset unique boolean 
        atomGroups = atomGroups.append(groupDict, ignore_index=True)
        unique = True
    
    #Summary statistics for the current dataframe:
     
    #Number of distinct groups 
    groups = atomGroups['Group number'].tolist()
    groups = np.asarray(groups)
    numGroups = len(np.unique(groups))
     
    #Number of atoms and subjects per group
    numAtoms_list = []
    numSubs_list = []
     
    for un in np.unique(groups):
        numAtoms = len(np.where(groups==un)[0])
        numAtoms_list.append(numAtoms)
     
        groupRows = atomGroups[atomGroups['Group number']==un]
        sub_list = np.asarray(groupRows['Subject ID'].tolist())
        numSubs = len(np.unique(sub_list))
        numSubs_list.append(numSubs)
     
    numAtoms_list = np.asarray(numAtoms_list)
    meanAtoms = np.mean(numAtoms_list)
    stdAtoms = np.std(numAtoms_list)
     
    print("Number of groups:")
    print(numGroups)
     
    print("Average number of atoms per group:")
    print(str(meanAtoms) + " +/- " + str(stdAtoms))
     
    groupSummary = pd.DataFrame(columns=['Group Number', 'Number of Atoms','Number of Subjects'])
    groupSummary['Group Number'] = np.unique(groups)
    groupSummary['Number of Atoms'] = numAtoms_list
    groupSummary['Number of Subjects'] = numSubs_list
     
    #Save group summary dataframe
    outputDir = '/media/NAS/lpower/CSC/results/u_' + str(u_thresh) + '_psd_' + str(psd_thresh) + '_groupSummary.csv'
    groupSummary.to_csv(outputDir)
     
    #Save atomGroups to dataframe
    outputDir = '/media/NAS/lpower/CSC/results/u_' + str(u_thresh) + '_psd_' + str(psd_thresh) + '_atomGroups.csv'
    atomGroups.to_csv(outputDir)


