import pandas as pd
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
import seaborn as sbn 
import statsmodels.formula.api as smf

dataDir = '/Users/lindseypower/Dropbox/PhD/Research Question 1A - Beta Event Detection/Data/'
taskAtomsFile = dataDir + 'taskClusters_allData.csv'
taskAtoms = pd.read_csv(taskAtomsFile)

#Function to calculate linear or quadratic regression 
def linear_regression(coef, clusterData, ctr, cluster, fig, ax, color, order):
        
    #Calculate regression
    x = clusterData['age']
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
    sbn.regplot(x="age", y=coef,data=clusterData, ax=ax[ctr], order=order, marker='.', color=color)
    ax[ctr].set_title(cluster)
    ax[ctr].set_ylabel(None)
    ax[ctr].set_xlabel(None)
    ax[ctr].set_ylim((-0.1,0.1))
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
        pred = coef['linear'][0]*subject['age'] + coef['intercept'][0]
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
        pred = coef['quadratic'][0]*subject['age']**2 + coef['linear'][0]*subject['age'] + coef['intercept'][0]
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

def compare_models(val, clusterData):
    # total num. of participants to analyze
    N = len(clusterData)
    
    #Calculate linear regression for current ROI
    x = clusterData['age']
    y = clusterData[val]
    
    # calculate linear  [ y = a + bx ] and quadratic  [ y = a + bx +cx^2 ]  model coefficients
    l_fit_coefficients = calc_l_fit_coefficients(x, y)
    q_fit_coefficients = calc_q_fit_coefficients(x, y)
    
    # caluclate chi^2 for each model (chi_sq_q < chi_sq_l, Always)
    chi_sq_l, list_of_w_l = calc_chi_square_l(clusterData, l_fit_coefficients, val)
    chi_sq_q, list_of_w_q = calc_chi_square_q(clusterData, q_fit_coefficients, val)
    
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
    
### MAIN:  Conduct linear and quadratic regression ######
variable = 'PostRate'
ctr = 0
fig, ax = plt.subplots(2,4)
ax = ax.reshape(-1)

#X position
for cluster in taskAtoms['Group number'].drop_duplicates().tolist():    
    print(cluster)
    clusterData = taskAtoms[taskAtoms['Group number']==cluster]  
    model = compare_models(variable, clusterData)
    if model == 'linear':
        order = 1
        color = 'b'
    elif model == 'quadratic':
        order = 2
        color = 'g'
    linear_regression(variable, clusterData, ctr, cluster, fig, ax, color=color, order=order)
    ctr = ctr+1
plt.show()
