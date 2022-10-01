import pandas as pd
import numpy as np

dataDir = '/Users/lindseypower/Dropbox/PhD/Research Question 1A - Beta Event Detection/Data/'
taskAtomsFile = dataDir + 'taskClusters_allData.csv'
allAtomsFile = dataDir + 'atomData_v2.csv'

#Read in all task-related atoms data and take only unique subjects 
taskAtoms = pd.read_csv(taskAtomsFile)

allAtoms = pd.read_csv(allAtomsFile)
allSubs = allAtoms.drop_duplicates(subset=['subject_id'])

#Calculate number of subjects in each age bin in the whole group
age_mins = [18,25.1,32.2,39.3,46.4,53.5,60.6,67.7,73.8,81.9]
percents = []
for age in age_mins:
    currSubs = allAtoms[allAtoms['age']>=age][allAtoms['age']<age+7.1]
    uniqueSubs = currSubs.drop_duplicates(subset=['subject_id'])
    percent = len(uniqueSubs)/len(allSubs)
    percents.append(percent)

#Dataframe to hold real and theoretical distributions of data for each cluster 
output_df = pd.DataFrame(columns = ['Cluster num', 'Age bin', 'Theoretical subjects', 'Real subjects', 'Chi-squared'])
demo_df = pd.DataFrame(columns = ['Cluster num', 'Age', 'Sex'])

#Calculate real and theoretical number of subjects per age bin and add to dataframe
for cluster in taskAtoms['Group number'].drop_duplicates().tolist():
    count = 0
    clusterData = taskAtoms[taskAtoms['Group number']==cluster]
    clusterSubs = clusterData.drop_duplicates(subset=['subject_id'])   
    for age in age_mins:
        currSubs = clusterData[clusterData['age']>=age][clusterData['age']<age+7.1]
        uniqueSubs = currSubs.drop_duplicates(subset=['subject_id'])       
        numSubs = len(uniqueSubs)
        theorSubs = percents[count]*len(clusterSubs)
        chisq = (numSubs - theorSubs)*(numSubs - theorSubs)/theorSubs
        
        output_df = output_df.append({'Cluster num': cluster, 'Age bin': age, 'Theoretical subjects': theorSubs, 
                                    'Real subjects': numSubs, 'Chi-squared': chisq}, ignore_index=True)
        
        curr_df = pd.DataFrame(columns = ['Cluster num', 'Age', 'Sex'])
        curr_df['Age'], curr_df['Sex'], curr_df['Cluster num'] = uniqueSubs['age'], uniqueSubs['sex'], cluster
        demo_df = demo_df.append(curr_df)
            
          
        count = count+1 
        
    mean_age = np.mean(np.asarray(clusterSubs['age'].tolist()))
    sd_age = np.std(np.asarray(clusterSubs['age'].tolist()))
    print(cluster)
    print(mean_age)
    print(sd_age)
    uniqueSubs = clusterData.drop_duplicates(subset=['subject_id'])   
    print(len(uniqueSubs))
    
output_file = dataDir +'age_demographics_data.csv'
output_df.to_csv(output_file)

#Repeat process for male and female
#Calculate number of subjects in each age bin in the whole group
sexes = ['MALE', 'FEMALE']
percents = []
for sex in sexes:
    currSubs = allAtoms[allAtoms['sex']==sex]
    uniqueSubs = currSubs.drop_duplicates(subset=['subject_id'])
    percent = len(uniqueSubs)/len(allSubs)
    percents.append(percent)

#Dataframe to hold real and theoretical distributions of data for each cluster 
output_df = pd.DataFrame(columns = ['Cluster num', 'Sex', 'Theoretical subjects', 'Real subjects', 'Chi-squared'])

#Calculate real and theoretical number of subjects per age bin and add to dataframe
for cluster in taskAtoms['Group number'].drop_duplicates().tolist():
    count = 0
    clusterData = taskAtoms[taskAtoms['Group number']==cluster]
    clusterSubs = clusterData.drop_duplicates(subset=['subject_id'])   
    for sex in sexes:
        currSubs = clusterData[clusterData['sex']==sex]
        uniqueSubs = currSubs.drop_duplicates(subset=['subject_id'])
        numSubs = len(uniqueSubs)
        theorSubs = percents[count]*len(clusterSubs)
        chisq = (numSubs - theorSubs)*(numSubs - theorSubs)/theorSubs
        
        output_df = output_df.append({'Cluster num': cluster, 'Sex': sex, 'Theoretical subjects': theorSubs, 
                                    'Real subjects': numSubs, 'Chi-squared': chisq}, ignore_index=True)
          
        count = count+1 
    
output_file = dataDir +'sex_demographics_data.csv'
output_df.to_csv(output_file)

#Demographic plots       
ax6 = sbn.violinplot(x="sex", y="age", data=allSubs)
# Shrink current axis by 20%
box = ax6.get_position()
ax6.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax6.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
    
