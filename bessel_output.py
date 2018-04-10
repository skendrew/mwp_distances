
# coding: utf-8

# In this notebooks I will examine the output of the Bessel distances to the HII regions, compare with Anderson et al results, and make some plots.

# In[47]:


#%matplotlib inline
import numpy as np
import astropy.io.ascii as ascii
import matplotlib.pyplot as plt
import os
import glob
import pdb


# Read in the HII region and bubble files, for comparison.

# In[48]:


mwpf = '/Users/kendrew/milkywayproj/distances/mwp_dr1L_withprob.csv'
#hiif = '/Users/kendrew/milkywayproj/anderson_HII/anderson14_wise/wise_HIIcat_web.csv'
hiif = '/Users/kendrew/milkywayproj/anderson_HII/wise_hii_V2.0.csv'
colnames = ['WISEname', 'catalog', 'GLong', 'GLat', 'radius', 'HIIregion', 'membership', 'vlsr', 'author', 'vlsr_mol', 'kdar', 'dist', 'err_dist', 'dist_method', 'rgal', 'z', 'glimpse8', 'wise12', 'wise22', 'mips24', 'higal70', 'higal160', 'hrds3cm', 'magpis20cm', 'vgps21cm']

mwp = ascii.read(mwpf)
hii = ascii.read(hiif)


# In[49]:


hii.keys()


# In[50]:


mwp.keys()


# ### Making plots from the output
# 
# In this section I'll run through all the output folders and make a plot showing the combined PDF and the individual components.

# In[51]:


basedir = '/Users/kendrew/milkywayproj/distances/'
os.chdir(basedir)
outdir = 'bayes_output/'
os.chdir(outdir)

# create a list of the directories
dirs = glob.glob('1G*')
print('found {0} directories'.format(len(dirs)))


pdb.set_trace()

# In[55]:


for d in dirs:
    
    flist = (glob.glob(d+'/*pdf.dat'))
    
    cols = ['dist', 'prob']
    colors = ['yellowgreen', 'gold', 'skyblue', 'coral', 'turquoise']
    
    plt.figure(figsize=[6,4])
    
    for f, c in zip(flist, colors):
        a = 1.
        if 'spiral' in f:
            cols = cols + ['arm']
        if 'final' not in f:
            a = 0.5
        
        
        lab = str.split(f, '_')[1]
        t = ascii.read(f, names=cols, comment='!')
        plt.plot(t['dist'], t['prob'], color=c, alpha=a, label=lab)
        plt.xlabel('distance (kpc)')
        plt.ylabel('probability density (kpc$^{-1}$)')
        plt.legend(loc='best')
        plt.title(d, fontsize='large')
        
    outfile = 'plots/' + d + '_pdf_plots.png'
    plt.savefig(outfile)
    plt.close()
    


os.chdir('..')


