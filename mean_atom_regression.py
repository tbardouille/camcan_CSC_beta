import pandas as pd
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
import seaborn as sbn 
import statsmodels.formula.api as smf

dataDir = '/Users/lindseypower/Dropbox/PhD/Research Question 1A - Beta Event Detection/Data/'
taskAtomsFile = dataDir + 'taskClusters_allData.csv'
meanAtomsFile = dataDir + 'df_mean_atom.csv'

#Function to convert string in dataframe to array 
def toArray(string):
    lst = string.split()
    if lst[0] == "[":
        lst = lst[1:]
    if lst[-1] == "]":
        lst = lst[:-1]
    num_list = []
    for x in lst:
        x = x.replace("[","")
        x = x.replace("]","")
        x = float(x)
        num_list.append(x)

    array = np.asarray(num_list)
    
    return array

#Function to calculate linear regression 
def linear_regression(coef, clusterData, ctr, cluster, fig, ax, color, order):
        
    #Calculate regression
    x = clusterData['Subject age']
    y = clusterData[coef]
    coefs = np.polyfit(x, y, order)
    df=pd.DataFrame(columns=['y','x'])
    df['x'] = x
    df['y'] = y
    model = np.poly1d(coefs) 
    results = smf.ols(formula ='y ~ model(x)', data=df).fit()
    p_val_lin = results.pvalues.loc['model(x)']
    print(results.t_test(['model(x)']))  
    print('p-val:')
    print(p_val_lin)
    print('coefs: ')
    print(coefs)
        
    #Plot Regression
    sbn.regplot(x="Subject age", y=coef,data=clusterData, ax=ax[ctr], order=order, marker='.', color=color)
    ax[ctr].set_title(cluster)
    ax[ctr].set_ylabel(None)
    ax[ctr].set_xlabel(None)
    ax[ctr].set_ylim((0.2,1.2))
    ax[ctr].set_xlim((15,90))
    ax[ctr].grid(True)
    
#Functions to calculate the linear and quadratic fits and conduct statistics to compare them
def calc_l_fit_coefficients(x, y):
	
	degree = 1
	coefs = np.polyfit(x, y, degree)
	linear_parameters = ['characteristic', 'linear', 'intercept']
	df_linear_fit = pd.DataFrame(columns=linear_parameters)
	linear_values = ['exponent', coefs[0], coefs[1]]
	dictionary = dict(zip(linear_parameters, linear_values))
	df_linear_fit = df_linear_fit.append(dictionary, ignore_index=True)	

	return df_linear_fit


def calc_q_fit_coefficients(x, y):
	
	degree = 2
	coefs = np.polyfit(x, y, degree)
	quadratic_parameters = ['characteristic', 'quadratic', 'linear', 'intercept']
	df_quadratic_fit = pd.DataFrame(columns=quadratic_parameters)
	quadratic_values = ['exponent', coefs[0], coefs[1], coefs[2]]
	dictionary = dict(zip(quadratic_parameters, quadratic_values))
	df_quadratic_fit = df_quadratic_fit.append(dictionary, ignore_index=True)		

	return df_quadratic_fit


def calc_chi_square_l(df, coef, val):
    chi_square = 0
    list_of_variances = []
    for index, subject in df.iterrows():
        real = subject[val]
        pred = coef['linear'][0]*subject['Subject age'] + coef['intercept'][0]
        w = ((real-pred)**2)/pred
        var = real-pred
        chi_square = w + chi_square
        list_of_variances.append(var)
        
    return chi_square, list_of_variances

def calc_chi_square_q(df, coef, val):
    chi_square = 0
    list_of_variances = []
    for index, subject in df.iterrows():
        real = subject[val]
        pred = coef['quadratic'][0]*subject['Subject age']**2 + coef['linear'][0]*subject['Subject age'] + coef['intercept'][0]
        w = ((real-pred)**2)/pred
        var = real-pred
        list_of_variances.append(var)
        #print(w)
        
        chi_square = w + chi_square
        
    return chi_square, list_of_variances

def calc_F_stat(N, chi_sq_l, chi_sq_q):

	m = 3 # m = num. of adjustable parameters in quad model (i.e. a,b,c)
	p = 1 # p = difference in num. of adjustable parameters between the two models (i.e. we fixed one parameter, a=0)
	f1 = p # degrees of freedom
	f2 = N-m # degrees of freedom 
	num = (chi_sq_l-chi_sq_q)/p 
	den = chi_sq_q/(N-m)
	F = num/den

	return F, f1, f2

def compare_models(coef, clusterData):
    # total num. of participants to analyze
    N = len(clusterData)
    
    #Calculate linear regression for current ROI
    x = clusterData['Subject age']
    y = clusterData[coef]
    
    # calculate linear  [ y = a + bx ] and quadratic  [ y = a + bx +cx^2 ]  model coefficients
    l_fit_coefficients = calc_l_fit_coefficients(x, y)
    q_fit_coefficients = calc_q_fit_coefficients(x, y)
    
    # caluclate chi^2 for each model (chi_sq_q < chi_sq_l, Always)
    chi_sq_l, list_of_w_l = calc_chi_square_l(clusterData, l_fit_coefficients, coef)
    chi_sq_q, list_of_w_q = calc_chi_square_q(clusterData, q_fit_coefficients, coef)
    
    clusterData['Quad_devs'] = list_of_w_q
    clusterData['Lin_devs'] = list_of_w_l
    
    print('chi-square linear:')
    print(chi_sq_l)
    print('chi-square quadratic:')
    print(chi_sq_q)
    
    #calc F-statistic and d.o.f
    F, f1, f2 = calc_F_stat(N, chi_sq_l, chi_sq_q)
    
    print('F-stat:')
    print(F)
    
    # test the null hypothesis
    if F>6.635000:
        appropriate_model = 'quadratic'
    if F<6.635000:
        appropriate_model = 'linear'
    print('appropriate model:')
    print(appropriate_model)
    
    return appropriate_model

###### Main ######

### Calculate correlation coefficients and format dataframe ###
   
#Read in taskAtom and meanAtom dataframes 
taskAtoms = pd.read_csv(taskAtomsFile)
meanAtoms = pd.read_csv(meanAtomsFile)

#Dataframe to store outputted correlation coefficients 
output_df = pd.DataFrame(columns = ['Cluster num', 'Subject id', 'Subject age', 'Subject sex','Atom num', 'u coef', 'v coef'])

for cluster in taskAtoms['Group number'].drop_duplicates().tolist():
    print(cluster)
    #Get information about each atom in the cluster
    clusterData = taskAtoms[taskAtoms['Group number']==cluster]
    subjects = clusterData['subject_id'].tolist()
    ages = clusterData['age'].tolist()
    sexes = clusterData['sex'].tolist()
    atom_ids = clusterData['atom_id'].tolist()
    u_data = clusterData['u_hat'].tolist()
    v_data = clusterData['v_hat'].tolist()
    
    #get information about the mean atom for the cluster
    clusterMean = meanAtoms[meanAtoms['label']==cluster]
    u_mean = toArray(clusterMean['u_hat'].tolist()[0])
    v_mean = toArray(clusterMean['v_hat'].tolist()[0])
    
    psd = np.abs(np.fft.rfft(v_mean, n=256)) ** 2
    peak_psd = np.where(psd==np.max(psd))[0][0]
    frequencies = np.linspace(0,150.0/2,len(psd))
    peak_freq = frequencies[peak_psd]
    print(peak_freq)
    
    #For each atom, find the correlation of the u and v vector to the mean atom's u and v vector 
    for row in range(0,len(clusterData)):
        u_curr = toArray(u_data[row])
        v_curr = toArray(v_data[row])
        
        #Correlation coefficient of u 
        u_coef = abs(np.corrcoef(u_curr, u_mean)[0,1])
        
        #max cross correlation coefficient of v 
        v_coef = np.max(ss.correlate(v_curr, v_mean))
        
        #Append row to dataframe 
        output_df = output_df.append({'Cluster num': cluster, 'Subject id': subjects[row], 'Subject age': ages[row], 
                                    'Subject sex': sexes[row], 'Atom num': atom_ids[row], 'u coef': u_coef,
                                    'v coef': v_coef}, ignore_index=True)

#Save dataframe to file
output_file = dataDir +'mean_correlation_data.csv'
output_df.to_csv(output_file)

### Conduct linear regression ###
ctr = 0
fig, ax = plt.subplots(2,4)
ax = ax.reshape(-1)

#U coefficient
for cluster in taskAtoms['Group number'].drop_duplicates().tolist():    
    clusterData = output_df[output_df['Cluster num']==cluster]  
    model = compare_models('u coef', clusterData)
    if model == 'linear':
        order = 1
        color = 'b'
    elif model == 'quadratic':
        order = 2
        color= 'g'
    linear_regression('u coef', clusterData, ctr, cluster, fig, ax, color=color, order=order)
    ctr = ctr+1
plt.show()

#V coefficient
ctr = 0
fig, ax = plt.subplots(2,4)
ax = ax.reshape(-1)

for cluster in taskAtoms['Group number'].drop_duplicates().tolist():    
    clusterData = output_df[output_df['Cluster num']==cluster]  
    model = compare_models('v coef', clusterData)
    if model == 'linear':
        order = 1
        color = 'b'
    elif model == 'quadratic':
        order = 2
        color = 'g'
    linear_regression('v coef', clusterData, ctr, cluster, fig, ax, color=color, order=order)
    ctr = ctr+1
plt.show()
        
        