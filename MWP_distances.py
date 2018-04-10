
# coding: utf-8

# In[135]:

#get_ipython().magic(u'matplotlib inline')

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import astropy.io.ascii as ascii
import astropy.io.votable as votable
import astropy.units as u
from astropy.coordinates import SkyCoord, search_around_sky, match_coordinates_sky
from astropy.coordinates import ICRS, Galactic
from astropy.table import Table, Column
from astropy.wcs import WCS
from astropy.io import fits
import scipy.spatial as spatial
from sklearn.neighbors import KDTree
import datetime
import shutil
import pdb
import glob
import os


plt.ion()

plt.close('all')

#==================================================
def call_distance(path):
#    '''This function will take the path and run the bayesian distance estimator on the source file inside it.
#    '''
    wdir = os.getcwd()
    os.chdir(path)
    os.system("./bayes_distance")
    os.system("rm -f fort.*")
    os.chdir(wdir)
    print('Bayesian distance estimator complete!')
    
#=================================================
def distance_cleanup(names, outdir):
    # scan through all the source names, create diretories with those names if don't already exist. 
    # search for all output files in the bessel_bayes directory and move over into its own directory
    #pdb.set_trace()
    wdir = os.getcwd()
    os.chdir(outdir)
    for name in names:
        outfiles = glob.glob(name[:8]+'*')

        if not os.path.isdir(wdir+'/bessel_output/'+name):
            os.mkdir(wdir+'/bessel_output/'+name)
    
        for file in outfiles:
            shutil.move(file, wdir+'/bessel_output/'+name+'/'+file)
    
    os.chdir(wdir)
    print('All files moved!')

#=================================================
       


# In this notebook I'm going to try and prepare a bubble catalogue for Bayesian distance determination using the Menten et al's method from the Bessel survey. Broad steps:
# 
# 1. match the bubbles to an HII region with velocities
# 2. write out the list of bubbles with good velocity matches to a text file
# 
# Data I'm using:
# * the large bubbles with Brut probabilities 
# * the WISE HII region catalogue
# * the inputs from Bessel (parallaxes)
# 

# ### Step 1: Read in data files 

# In[77]:

def create_xmatch(outfile=False, plot=False):
    
    '''This function will cross-match the mwp and HII region files, within the bubble radius. Two optional parameters:
    
    outfile = True/False determines whether the table will be written to file. BE CAREFUL this will overwrite the inp file for the code, including the modifications for the duplicate entries!
    plot = True/False determines whether I want to create a bunch of plots. can be annoying or useful.
    '''
    
    
    mwpf = '/Users/kendrew/milkywayproj/mwp_distances/mwp_dr1L_withprob.csv'
    #hiif = '/Users/kendrew/milkywayproj/anderson_HII/anderson14_wise/wise_HIIcat_web.csv'
    hiif = '/Users/kendrew/milkywayproj/anderson_HII/wise_hii_V2.0.csv'

    mwp = ascii.read(mwpf)
    hii = ascii.read(hiif)


    # Convert longitude coordinate to negatives
    mwp_neglon = mwp['GLON'] >= 180.
    hii_neglon = hii['GLong_deg'] >= 180.
    mwp['GLON'][mwp_neglon] = mwp['GLON'][mwp_neglon] - 360.
    hii['GLong_deg'][hii_neglon] = hii['GLong_deg'][hii_neglon] - 360.

    # Let's select just the HII regions that have a velocity measurement. 
    hiivel = hii[~hii['vlsr_kms'].mask]
    print('Number of HII regions with a velocity measurement: {0} (out of {1} total)' .format(len(hiivel), len(hii)))
    
    # narrow down the HII region catalogue to the area of coverage of the bubbles. makes things a bit faster
    area = (np.abs(hiivel['GLong_deg']) <= 65.) & (np.abs(hiivel['GLat_deg']) <= 1.) 
    hiivel = hiivel[area]
    hiivel.sort(['GLong_deg'])
    #len(hiivel)

    # Now also throw out those with multiple velocities
    tmp = np.char.find(hiivel['vlsr_kms'], ';')
    hiivel = hiivel[tmp<0]

    # Now define the coordinate columns as SkyCoord objects so can cross-match
    mwp_coord = SkyCoord(l=mwp['GLON'], b=mwp['GLAT'], unit=u.deg, frame="galactic")
    hiivel_coord = SkyCoord(l=hiivel['GLong_deg'], b=hiivel['GLat_deg'], unit=u.deg, frame="galactic")
    hii_coord = SkyCoord(l=hii['GLong_deg'], b=hii['GLat_deg'], unit=u.deg, frame="galactic")

    # Now search around each bubble to find an HII region within its radius
    idx, sep2d, sep3d = match_coordinates_sky(hiivel_coord, mwp_coord, nthneighbor=1,storekdtree='kdtree_sky')
    matches = (sep2d <= mwp['Reff'][idx]*u.arcmin)
    bub_matches = idx[matches]
    
    
    if plot:
        fig = plt.figure(figsize=[12,4])
        plt.scatter(mwp['GLON'], mwp['GLAT'], marker='o', facecolor='None', edgecolor='b', label='bubbles')
        plt.scatter(hii['GLong_deg'], hii['GLat_deg'], marker='x', color='r', label='HII regions (all)')
        plt.title('Bubble and HII region locations')
        plt.legend()

        plt.figure(figsize=(12, 4))
        plt.scatter(mwp['GLON'][bub_matches], mwp['GLAT'][bub_matches], marker='o', facecolor='None', edgecolor='b', label='matched bubbles')
        plt.scatter(hiivel['GLong_deg'][matches], hiivel['GLat_deg'][matches], marker='x', c='r', label='matched HII regions (with vlsr)')
        plt.legend()
        fig.show()

    # Remember the marker sizes don't represent the bubble sizes in this plot so it's okay to have an HII region outside the marker - can be a very large bubble
    mwp_matched = mwp[bub_matches]
    hii_matched = hiivel[matches]
    len(mwp_matched), len(hii_matched)


    # ### Some statistics for the matches

    # Let's look at the **distances between bubbles and their matched HII regions**
    rbins = np.arange(0., 3.0, 0.1)

    fig = plt.figure()
    plt.hist(sep2d[matches].to(u.arcmin)/mwp_matched['Reff'], bins=rbins, histtype='step', lw=2)
    fig.show()

    # What is the median separation (absolute) between matched bubbles and HII regions?
    med_sep = np.median(sep2d[matches])
    med_sep.to(u.arcsec)


    # Ditto as a fraction of their effective radii?
    med_sep_reff = np.median((sep2d[matches].to(u.arcmin)).value/mwp_matched['Reff'])


    # What are the **_velocities_** of the HII regions matched to bubbles? First throw out those with multiple velocities.
    vlsr_matched = np.zeros(len(hii_matched), dtype=np.float)
    vlsr_all = np.zeros(len(hiivel), dtype=np.float)
    for i in range(len(hii_matched)):
        vlsr_matched[i] = hii_matched['vlsr_kms'][i].astype(np.float)
    for i in range(len(hiivel)):
        vlsr_all[i] = hiivel['vlsr_kms'][i].astype(np.float)

    vbins = np.arange(np.min(vlsr_all), np.max(vlsr_all), 5.)
    vhist = plt.hist(vlsr_matched, bins=vbins, histtype='step', lw=2, color='r', label='matched', cumulative=True, normed=True)
    vhist2 = plt.hist(vlsr_all, bins=vhist[1], histtype='step', lw=2, color='b', label='all', cumulative=True, normed=True)
    plt.title('Distribution of HII region velocities')
    plt.legend(loc=2)


    # Those are some identical-looking distributions! Looks like bubbles are not preferentially matched to particular velocities

    # **Are we preferentially matching in certain longitude regions?**
    if plot:
        lonbins = np.arange(-65., 65., 5.)
        plt.figure(figsize=[12,4])
        plt.hist(hiivel['GLong_deg'], bins=lonbins, histtype='step', lw=2, color='b', label='all HII regions', normed=True)
        plt.hist(hii_matched['GLong_deg'], bins=lonbins, histtype='step', lw=2, color='r', label='matched HII regions', normed=True)
        plt.legend(loc=2)
        plt.xlabel('galactic longitude')
        plt.ylabel('distribution (normed)')

    # maybe a slight difference around l = 20-30 and around the GC...?

    # **Longitude-velocity plot**
    if plot:
        plt.figure(figsize=[12,4])
        plt.scatter(hiivel['GLong_deg'], hiivel['vlsr_kms'], marker='x', color='b', label='all HII regions')
        plt.scatter(hii_matched['GLong_deg'], hii_matched['vlsr_kms'], marker='x', color='r', label='matched HII regions')
        plt.legend(loc=3)
        plt.xlim([70, -70])
        plt.xlabel('longitude (deg)')
        plt.ylabel('VLSR (km/s)')



    # **How do the probabilities of the matched v unmatched bubbles compare?**
    if plot:
        plt.figure(figsize=[6,6])
        prob_bins = np.arange(0.0, 1.1, 0.05)
        plt.hist(mwp['prob'], bins=prob_bins, histtype='step', lw=2, color='r', label='all', normed=True, cumulative=True)
        plt.hist(mwp_matched['prob'], bins=prob_bins, histtype='step', lw=2, color='g', label='matched', normed=True, cumulative=True)
        plt.hist(mwp[~bub_matches]['prob'], bins=prob_bins, histtype='step', lw=2, color='b', label='no match', normed=True, cumulative=True)
        plt.xlabel('brut probability')
        plt.ylabel('distribution')
        plt.ylim([0., 1.1])
        plt.legend(loc=2)

    # ### Output creation

    # Need to identify duplicates and tag them somehow so the distance code doesn't overwrite the output for all of them. Want to keep the distinct matches.
    # NB need to pull out the names into a separate array, then change in numpy, and reinsert - as the string column in the table has some weird behaviour and I can't get it to work. Seems known - see https://github.com/astropy/astropy/issues/5551
    # No time to dig into the details though.
    # OK so I HAD to look into the details and seems like this string truncation is a numpy behaviour ARGH. I'm goignto have to go in manually and change the entries in the file. This sucks.


    fout = '../bessel_bayes/sources_info.inp'
    comm_string = 'created on {0} -- SK using v2.0 HII region file' .format(str(datetime.datetime.now()))
    p_far = 0.5 + np.zeros(len(mwp_matched), dtype=np.float)
    #len_t = 10
    t = Table([mwp_matched['MWP'], mwp_matched['GLON'], mwp_matched['GLAT'], hii_matched['vlsr_kms'], p_far, hii_matched['wise_Name']], names=('!MWP', 'GLON', 'GLAT', 'Vlsr', 'P(far)', 'WISE name'), 
        meta={'comments': [comm_string]})
    
    if outfile:
        ascii.write(t, fout, delimiter='\t', comment='!')

    return t
#==================================================================================================================

# Then we need to call the fortran code 
rundir = '/Users/kendrew/milkywayproj/bessel_bayes/'
call_distance(rundir)

# Then I need to do a massive clean-up operation. Read in the input file to generate the directories
f = '../bessel_bayes/sources_info.inp'
inp = ascii.read(f, comment='!', names=['!MWP',	'GLON', 'GLAT',	'Vlsr', 'P(far)', 'WISE name'], include_names=['!MWP'])
distance_cleanup(inp['!MWP'], rundir)













