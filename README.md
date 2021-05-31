Steps to complete beta burst CSC analysis

We expect that atoms representing sensorimotor beta will have more activation in the pre-movement
interval compared to movement interval. Thus, they should have a large, negative activation change
and t-stat.

1. Run camcan_process_to_evoked_parallel.py to pre-process all participant data.
	This step will:
		- read in raw data from CamCAN that has tSSS applied including 
				trans to default head position
		- find all button press events
		- discard presses with RT>1000ms or time to previous button press < 3000ms
		- epoch to 3.4 s centred on the button press
		- apply ICA to remove artifacts based on magnitude and temporal relation
				 to ECG/EOG

2.. Run check_TFR.py to see if this participant has a nice, juicy beta suppression

3. Run run_CSC_grad.py to calculate the CSC results
	- duration is set to 500 ms in this analysis to be sensitive to beta bursts
	- this should be a better option than 1000ms because median beta burst 
			duration is ~200ms
	- 25 atoms
	- atom duration = 500 ms
	- band-pass at 2-45Hz
	- sample rate dropped to 300 Hz

4. Run plot_CSC_grad.py to plot the results of the CSC analysis
	- this will plot the:
			- atom timecourses (V-vectors) 
			- atom topographies (U-vectors)
			- atom activation vectors (Z-vectors) overlaid per epoch
			- aggregate movement interval activation - aggregate pre-movement interval activation
			- t-test (across trials) of [movement interval activation - pre-movement interval activation]


