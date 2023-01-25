Convolutional sparse coding (CSC) was applied to the CamCAN dataset to detect repeating spatiotemporal atoms in individual participants. The atoms were then clustered across participants to form groups of highly similar atoms. Age-related trends in atom characteristics were then investigated within clusters. 

When using this code, please cite https://doi.org/10.1016/j.neuroimage.2022.119809

Follow the steps below to complete the CSC + clustering process:

1. Run ‘run_csc_parallel.py’:
- relies on functions from ‘utils_csc.py’ and ‘utils_plot.py’
- reads in the raw MEG data (from the sensorimotor task) for each participant in BIDS format
- preprocesses the data by applying max filter, epoching the data based on button press timing, and bandpass filtering between 2 and 45 Hz
- applies CSC to preprocessed data 
- saves CSC model info (e.g., u vector, v vector, activation vector) to pickle file
- fits a dipole for each atom using u vector information 
- calculates percent change between phases of movement 
- saves dipole and percent change info to ‘atomData.csv’

2. Run ‘correlation_clustering_singleSub.py’:
- clusters atoms within participants using the correlation coefficients of the u and v vectors 
- prints a list of participants who should be excluded because the majority of their atoms are the same (I.e., the variability in the signal is not properly captured)
- The outputted list should be copied and pasted into exclude_subs list variable in ‘correlation_clustering_allAtoms.py’

3. Run ‘correlation_clustering_allAtoms.py’:
- clusters atoms across participants using the correlation coefficients of the u and v vectors 
- saves a data frame with summary information about each resulting cluster (e.g., number of atoms, number of subjects) as ‘u_0.4_v_0.4_groupSummary.csv’
- saves a dataframe with the subject, atom number, and cluster number to which it was assigned as ‘u_0.4_v_0.4_atomGroups.csv’

4. Run ‘find_taskClusters.py’:
- reads in ‘atomData.csv’, ‘_groupSummary.csv’, and ‘_atomGroups.csv’ files 
- selects a set of ‘top’ (I.e., highly stereotypical) clusters as those with a high number of participants represented
- of those, selects task-related clusters as those with a task-related reduction in activity, and a focal dipole source, on average
- saves data frames containing summary information and characteristics of the clusters including ‘topClustersSummary.csv’, ‘taskClustersSummary.csv’, and ‘taskClusters_allData.csv’

Various statistical analyses and visualizations of these data can then be completed using the scripts available in the ‘visualization’ folder:

Visualizations: 
- similar_atoms.py: creates correlation matrices comparing atoms within participants
- mean_atom.py: creates a visual representation of the u and v vector for a representative atom for each cluster
- create_tfr.py: calculates and plots TFRs using the concatenated atom data from each cluster
- plot_dipole.py: plots a dipole on the template brain for each representative atom

Statistics:
- demographics.py: compares demographic distribution for each cluster to the overall demographic distribution for the whole camCAN dataset
- spatial_temporal_regression.py: conducts linear and quadratic regression analysis between various atom characteristics and participant age
- activation_histogram.py: calculates the distribution of atom activation values for different participant ages and conducts linear and quadratic regression between the mu/sigma values of the distribution and participant age
- mean_atom_regression.py: calculates the correlation between the representative atom and each atom in the clusters and regresses the correlation with age
- spatial_cluster_plot.py: plots the x, y, and z position of each atom’s dipole on a single plot with age-based colouring


