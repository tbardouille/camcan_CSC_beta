import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn 
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


dataDir = '/Users/lindseypower/Dropbox/PhD/Research Question 1A - Beta Event Detection/Data/'
taskAtomsFile = dataDir + 'taskClusters_allData.csv'
meanAtomsFile = dataDir + 'mean_dipole.csv'
allAtomsFile = dataDir + 'atomData_v2.csv'

def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


taskAtoms = pd.read_csv(taskAtomsFile)
meanAtoms = pd.read_csv(meanAtomsFile)
colours = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:pink','tab:grey']

fig, ax = plt.subplots(2,2)
ax = ax.reshape(-1)
count = 0
for cluster in taskAtoms['Group number'].drop_duplicates().tolist():
    print(cluster)
    
    #Plot x,z relationship in top left panel
    clusterData = taskAtoms[taskAtoms['Group number']==cluster]
    meanClusterData = meanAtoms[meanAtoms['Cluster Number']==cluster]
    #clusterData.plot.scatter(x='Dipole Pos x', y='Dipole Pos z', c= 'age', colormap='rainbow', ax=ax[0], s=[10])
    meanClusterData.plot.scatter(x='Dipole Pos x', y='Dipole Pos z', c=colours[count], ax=ax[0], marker='.', s=[20])
    confidence_ellipse(np.asarray(clusterData['Dipole Pos x'].tolist()), np.asarray(clusterData['Dipole Pos z'].tolist()), ax[0], edgecolor=colours[count])
    ax[0].set_ylim((-0.025,0.175))
    ax[0].set_xlim((-0.11,0.11))
    ax[0].set_xlabel('')
    #ax[0].set_title(cluster)
    
    #plot y,z relationship in top right panel
    clusterData = taskAtoms[taskAtoms['Group number']==cluster]
    meanClusterData = meanAtoms[meanAtoms['Cluster Number']==cluster]
    #clusterData.plot.scatter(x='Dipole Pos y', y='Dipole Pos z', c= 'age', colormap='rainbow', ax=ax[1], s=[10])
    meanClusterData.plot.scatter(x='Dipole Pos y', y='Dipole Pos z', c=colours[count], ax=ax[1], marker='.', s=[20])
    confidence_ellipse(np.asarray(clusterData['Dipole Pos y'].tolist()), np.asarray(clusterData['Dipole Pos z'].tolist()), ax[1], edgecolor=colours[count])
    ax[1].set_ylim((-0.025,0.175))
    ax[1].set_xlim((-0.11,0.11))
    ax[1].set_ylabel('')
    
    #plot x,y relationship in bottom left panel
    clusterData = taskAtoms[taskAtoms['Group number']==cluster]
    meanClusterData = meanAtoms[meanAtoms['Cluster Number']==cluster]
    #clusterData.plot.scatter(x='Dipole Pos x', y='Dipole Pos y', c= 'age', colormap='rainbow', ax=ax[2], s=[10])
    meanClusterData.plot.scatter(x='Dipole Pos x', y='Dipole Pos y', c=colours[count], ax=ax[2], marker='.', s=[20])
    confidence_ellipse(np.asarray(clusterData['Dipole Pos x'].tolist()), np.asarray(clusterData['Dipole Pos y'].tolist()), ax[2], edgecolor=colours[count])
    ax[2].set_ylim((-0.11,0.11))
    ax[2].set_xlim((-0.11,0.11))
    
    count = count + 1

custom_lines = [Line2D([0], [0], color=colours[0], lw=4),
                Line2D([0], [0], color=colours[1], lw=4),
                Line2D([0], [0], color=colours[2], lw=4),
                Line2D([0], [0], color=colours[3], lw=4),
                Line2D([0], [0], color=colours[4], lw=4),
                Line2D([0], [0], color=colours[5], lw=4),
                Line2D([0], [0], color=colours[6], lw=4)]

ax[3].legend(custom_lines, ['LO_alpha', 'RC_mu', 'RO_alpha', 'OP_alpha','LTA_alpha','LPreC_beta','LPostC_beta',])
plt.show()

#plot distribution of different clusters 
ax1 = sbn.violinplot(x="Group number", y="Peak Frequency", data=taskAtoms)
plt.show()

ax2 = sbn.violinplot(x="Group number", y="Dipole GOF", data=taskAtoms)
plt.show()

ax3 = sbn.violinplot(x="Group number", y="PreSum", data=taskAtoms)
ax3.set_ylim((-5e-9,2.5e-8))
plt.show()

ax4 = sbn.violinplot(x="Group number", y="MoveSum", data=taskAtoms)
ax4.set_ylim((-5e-9,2.5e-8))
plt.show()

ax5 = sbn.violinplot(x="Group number", y="PostSum", data=taskAtoms)
ax5.set_ylim(-5e-9,2.5e-8)
plt.show()

