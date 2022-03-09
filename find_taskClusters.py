import pandas as pd 

#Paths 
dataDir = '/Users/lindseypower/Dropbox/PhD/Research Question 1A - Beta Event Detection/'
allAtomsFile = dataDir + 'atomData_v2.csv'
groupSummaryFile = dataDir + 'u_0.4_v_0.4_groupSummary.csv'
atomGroupsFile = dataDir + 'u_0.4_v_0.4_atomGroups.csv'

#Read in all atom and mean atom dataframes 
allAtoms = pd.read_csv(allAtomsFile)
groupSummary = pd.read_csv(groupSummaryFile)
atomGroups = pd.read_csv(atomGroupsFile)

#Add group column to allAtoms dataframe 
allAtoms = pd.merge(allAtoms, atomGroups, on=['subject_id','atom_id'])  
allAtoms = allAtoms.fillna(0)  
allAtoms = allAtoms.replace('#NAME?',0)
allAtoms['Pre-Move Change'] = pd.to_numeric(allAtoms['Pre-Move Change'])
allAtoms['Post-Move Change'] = pd.to_numeric(allAtoms['Post-Move Change'])
allAtoms['Post-Pre Change'] = pd.to_numeric(allAtoms['Post-Pre Change'])

#Define variables
numSubjects = allAtoms.shape[0]/20
topCluster_threshold = 0.25 
gof_threshold = 0.9
change_threshold = 0.1
rebound_threshold = 0.1

#Find top clusters 
topClustersSummary = groupSummary[groupSummary['Number of Subjects']>topCluster_threshold*numSubjects]
topClusters = topClustersSummary['Group Number'].tolist()

#For each cluster, calculate the mean and standard dev of important features 
df_list = []
task_related = False
means_df  = pd.DataFrame(columns = ["Cluster Number","GOF Mean", "GOF Stdev", "Pre-Move Change Mean", "Pre-Move Change Stdev", 
                                    "Post-Move Change Mean","Post-Move Change Stdev", "Post-Pre Change Mean", "Post-Pre Change Stdev",
                                    "Task-Related"])
for cluster in topClusters: 
    curr_df = allAtoms[allAtoms['Group number'] == cluster]
    
    mean_gof = curr_df['Dipole GOF'].mean()
    sd_gof = curr_df['Dipole GOF'].std()
    mean_premove = curr_df['Pre-Move Change'].mean()
    sd_premove = curr_df['Pre-Move Change'].std()
    mean_postmove = curr_df['Post-Move Change'].mean()
    sd_postmove = curr_df['Post-Move Change'].std()
    mean_prepost = curr_df['Post-Pre Change'].mean()
    sd_prepost = curr_df['Post-Pre Change'].std()
    
    
    #Check if the current cluster should be consdiered a task-related cluster 
    if (mean_gof > gof_threshold and mean_postmove > change_threshold and mean_premove > change_threshold):
        df_list.append(curr_df)
        print(cluster)
        task_related = True
        
    #Append mean dataframe to mean dataframe
    means_df = means_df.append({"Cluster Number": cluster,"GOF Mean": mean_gof, "GOF Stdev": sd_gof, "Pre-Move Change Mean": mean_premove,
                                "Pre-Move Change Stdev": sd_premove, "Post-Move Change Mean": mean_postmove,"Post-Move Change Stdev": sd_postmove,
                                "Post-Pre Change Mean": mean_prepost, "Post-Pre Change Stdev": sd_prepost,"Task-Related": task_related}, ignore_index=True)
                                
    #Resst task-related boolean
    task_related = False
                                
                                
#This df has just the data in the task-related top clusters       
taskCluster_df = pd.concat(df_list)

#Save important data 
topClustersSummary.to_csv(dataDir + 'topClustersSummary.csv')
means_df.to_csv(dataDir + 'taskClustersSummary.csv')
taskCluster_df.to_csv(dataDir + 'taskClusters_allData.csv')


    





