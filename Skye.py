# -*- coding: utf-8 -*-

from subprocess import check_output, CalledProcessError
import numpy as np
import os
import scipy
import math
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
import matplotlib.colors as colors

reload(sys)
sys.setdefaultencoding('utf8')


# Main
def main():
    # Call susan skye for OPS, INV, and SS1
    calcSkye(1.0,3.0,45.0,'OPS-TTimes-Skygroup-Per-NoFSP.txt','skye-ops-susan-3.0sig-1.0day-45days-NoFSP.txt','skye-ops-susan-3.0sig-1.0day-45days-NoFSP-thresh.txt','OPS-skye-plots-susan-3.0sig-1.0day-45days-NoFSP.pdf')
    calcSkye(1.0,3.0,45.0,'INV-TTimes-Skygroup-Per-NoFSP.txt','skye-inv-susan-3.0sig-1.0day-45days-NoFSP.txt','skye-inv-susan-3.0sig-1.0day-45days-NoFSP-thresh.txt','INV-skye-plots-susan-3.0sig-1.0day-45days-NoFSP.pdf')
    calcSkye(1.0,3.0,45.0,'SS1-TTimes-Skygroup-Per-NoFSP.txt','skye-ss1-susan-3.0sig-1.0day-45days-NoFSP.txt','skye-ss1-susan-3.0sig-1.0day-45days-NoFSP-thresh.txt','SS1-skye-plots-susan-3.0sig-1.0day-45days-NoFSP.pdf')
    calcSkye(1.0,3.0,45.0,'SS2-TTimes-Skygroup-Per-NoFSP.txt','skye-ss2-susan-3.0sig-1.0day-45days-NoFSP.txt','skye-ss2-susan-3.0sig-1.0day-45days-NoFSP-thresh.txt','SS2-skye-plots-susan-3.0sig-1.0day-45days-NoFSP.pdf')
    calcSkye(1.0,3.0,45.0,'SS3-TTimes-Skygroup-Per-NoFSP.txt','skye-ss3-susan-3.0sig-1.0day-45days-NoFSP.txt','skye-ss3-susan-3.0sig-1.0day-45days-NoFSP-thresh.txt','SS3-skye-plots-susan-3.0sig-1.0day-45days-NoFSP.pdf')


    # Call calc bad tps transits for OPS, INV, SS1, and INJ
    calcBadTPSTCETransits(1.0,'OPS-TTimes-Skygroup-Per-NoFSP.txt','skye-ops-susan-3.0sig-1.0day-45days-NoFSP.txt','Skyline-Metric-OPS-susan.txt')
    calcBadTPSTCETransits(1.0,'INV-TTimes-Skygroup-Per-NoFSP.txt','skye-inv-susan-3.0sig-1.0day-45days-NoFSP.txt','Skyline-Metric-INV-susan.txt')
    calcBadTPSTCETransits(1.0,'SS1-TTimes-Skygroup-Per-NoFSP.txt','skye-ss1-susan-3.0sig-1.0day-45days-NoFSP.txt','Skyline-Metric-SS1-susan.txt')
    calcBadTPSTCETransits(1.0,'INJ-TTimes-Skygroup-Per-NoFSP.txt','skye-ops-susan-3.0sig-1.0day-45days-NoFSP.txt','Skyline-Metric-INJ-susan.txt')
    calcBadTPSTCETransits(1.0,'SS2-TTimes-Skygroup-Per-NoFSP.txt','skye-ss2-susan-3.0sig-1.0day-45days-NoFSP.txt','Skyline-Metric-SS2-susan.txt')
    calcBadTPSTCETransits(1.0,'SS3-TTimes-Skygroup-Per-NoFSP.txt','skye-ss3-susan-3.0sig-1.0day-45days-NoFSP.txt','Skyline-Metric-SS3-susan.txt')


def calcSkye(binwidth,sigma,periodcut,filenamein,filenameout,threshnameout,plotname,opt=None):
    """Compute times of transit clusters for each skygroup

    Inputs:
    -------------
    binwidth
        The bin width, in days, for Skye

    filenamein
        The name of the file containing the times of transit, skygroup, and period

    Returns:
    -------------
    None

    Output:
    ----------
    filenameout
        File containing times and skygroups, upon which any transits should be considered 'bad'
    threshnameout
        File continaing the thresholds for each skygroup
    plotname
        Plot showing histogram distribution and cutoff value
    """

    # General idea
    #
    # 1) Count the number of transits in the skygroup, and the number of bins
    #
    # 2) Divide, that's the rate
    #
    # 3) Multitple rate by S, that's the cutoff
    #
    # 4) Cut out outliers, and count new number of transits and new number of bins  (eliminate the bins that were cut out)
    #
    # 5) Multiply rate by S, get new cutoff
    #
    # 6) Repeat 4 and 5 for X times until sure it's converged.

    mint = 131.5
    maxt = 1591.0

    pdf_pages = PdfPages(plotname)
    outfile = open(filenameout, 'w')
    outfile2 = open(threshnameout, 'w')

    # Load from input file
    data = np.genfromtxt(filenamein, names=True, dtype=None)

    # Calculate gaptimes first
    ttimes = data[(data['ses']!=0)]['ttime']  # Select all times from all TCEs with all periods that have non-zero SES values (i.e., are not in gaps)
    hist = np.histogram(ttimes,bins=np.arange(mint, maxt + binwidth, binwidth))
    histvals = hist[0]
    bins = hist[1]
    bincenters = 0.5*(bins[1:]+bins[:-1])
    gaptimes = bincenters[histvals==0]  # If the histogram value is zero, then no TCE at any period ever made a transit there, so it's a gap, so record the times of the gaps

    # Now calculate real histogram, taking gaps into account, using P > periodcut
    ttimes = data[(data['period']>periodcut)]['ttime']     # Select TCEs that only have periods great than periodcut days
    hist = np.histogram(ttimes,bins=np.arange(mint, maxt + binwidth, binwidth))
    histvals = hist[0]
    bins = hist[1]
    bincenters = 0.5*(bins[1:]+bins[:-1])   # Compute the center time of each bin
    bincenterstmp = bincenters[~np.in1d(bincenters, gaptimes)]  # Only use non-gapped times for stats
    histvalstmp = histvals[~np.in1d(bincenters, gaptimes)]  # Only use histogram values corresponding to non-gapped times for stats
    nbins = len(bincenterstmp)
    ntrans = np.sum(histvalstmp)
    rate = 1.0*ntrans/nbins
    cutoff = rate + sigma*np.sqrt(rate)

    # Iterate on cutting out spikes to converge on cutoff value
    for i in range(0,5):
        bincenterstmp = bincenterstmp[([histvalstmp<=cutoff])]
        histvalstmp = histvalstmp[([histvalstmp<=cutoff])]
        nbins = len(bincenterstmp)
        ntrans = np.sum(histvalstmp)
        rate = 1.0*ntrans/nbins
        cutoff = rate + sigma*np.sqrt(rate)

    badtimes = bincenters[histvals>cutoff]

    outfile2.write("%3s %5.3f\n" % ('All',cutoff))

    # Make plot for all skygroups
    fig = plt.figure()
    fig.set_size_inches(11,8.5)
    plt.title("All Skygroups\n")
    plt.xlabel("Transit Time")
    plt.ylabel("Number")
    plt.axhline(y=cutoff, c="blue")
    plt.step(bincenters,histvals, color='k', rasterized=True)
    pdf_pages.savefig(fig, dpi=400)

    # Now loop over all the 84 skygroups
    for sg in range(1,85):

        # Calculate gaptimes first
        ttimes = data[np.logical_and(data['skygroup']==sg,data['ses']!=0)]['ttime']  # Select all times from all TCEs with all periods that have non-zero SES values (i.e., are not in gaps)
        hist = np.histogram(ttimes,bins=np.arange(mint, maxt + binwidth, binwidth))
        histvals = hist[0]
        bins = hist[1]
        bincenters = 0.5*(bins[1:]+bins[:-1])
        gaptimes = bincenters[histvals==0]  # If the histogram value is zero, then no TCE at any period ever made a transit there, so it's a gap, so record the times of the gaps

        # Now calculate real histogram, taking gaps into account, using P > periodcut
        ttimes = data[np.logical_and(data['skygroup']==sg,data['period']>periodcut)]['ttime']  # Select all transit times from TCEs on the given skygroup and with P > periodcut days
        hist = np.histogram(ttimes,bins=np.arange(mint, maxt + binwidth, binwidth))     # Calculate the histogram values from times mint to maxt with binning size binwidth
        histvals = hist[0]  # Save the histogram values (number in each bin)
        bins = hist[1]  # Save the bins (start and end times of each bin)
        bincenters = 0.5*(bins[1:]+bins[:-1])  # Compute the bin centers (so one bincenter for each histval)

        bincenterstmp = bincenters[~np.in1d(bincenters, gaptimes)]  # Only use non-gapped times for stats
        histvalstmp = histvals[~np.in1d(bincenters, gaptimes)]  # Only use histogram values corresponding to non-gapped times for stats
        nbins = len(bincenterstmp)
        ntrans = np.sum(histvalstmp)
        rate = 1.0*ntrans/nbins
        cutoff = rate + sigma*np.sqrt(rate)

        # Iterate on cutting out spikes to converge on cutoff value
        for i in range(0,5):
            bincenterstmp = bincenterstmp[([histvalstmp<=cutoff])]
            histvalstmp = histvalstmp[([histvalstmp<=cutoff])]
            nbins = len(bincenterstmp)
            ntrans = np.sum(histvalstmp)
            rate = 1.0*ntrans/nbins
            cutoff = rate + sigma*np.sqrt(rate)
        
        badtimes = bincenters[histvals>cutoff]

        outfile2.write("%3i %5.3f\n" % (sg,cutoff))

        for time in badtimes:
            outfile.write("%6.2f %2i\n" % (time,sg))

        # Make plot
        fig = plt.figure()
        fig.set_size_inches(11,8.5)
        plt.title("Skygroup %i\n" % (sg))
        plt.xlabel("Transit Time")
        plt.ylabel("Number")
        plt.axhline(y=cutoff, c="blue")
        plt.step(bincenters,histvals, color='k', rasterized=True)
        plt.ylim(ymin=0)
        pdf_pages.savefig(fig, dpi=400)


    outfile.close()
    outfile2.close()
    pdf_pages.close()



# Calculate which transits of which TCEs correspond to bad cadences on particular skygroups using the TPS times
def calcBadTPSTCETransits(binwidth,filenamein,skyefilename,outfilename):
    """Compute times of bad transits using Skye results

    Inputs:
    -------------
    binwidth
        The bin width, in days, for Skye

    filenamein
        The name of the file containing the times of transit, skygroup, and period

    skyefilename
        The output file from calcSkye

    outfilename
        The name of the output file containing TCEs and their bad transits


    Returns:
    -------------
    None

    Output:
    ----------
    File containing TCE IDs and transit times of bad transits due to Skye
    """

    maxt = 1600.0  # Maximum BKJD of the data

    # Read in the TCE data (name, period, epoch, and skygroup)
    data = np.genfromtxt(filenamein, names=True, dtype=None)

    # Read in the output file from calcSkye with the bad times and skygroups pairs
    skyedat = np.genfromtxt(skyefilename, dtype=None)

    # Open the output file for writing
    outfile = open(outfilename, 'w')

    # Find bad transit times
    for sg in range(1,85):    # Loop over all skygroups
        tmpdata = data[(data['skygroup']==sg)]    # Select current skygroup - speeds up execution
        tmpskyedat = skyedat[(skyedat['f1']==sg)]    # Select current skygroup - speeds up execution
        for i in tmpskyedat:    # Loop through each bad transit time / skygroup pair identified by Skye
            for j in tmpdata:    # Loop through each TCE's transit times
                if(j['ttime'] > i[0]-binwidth/2.0 and j['ttime'] < i[0]+binwidth/2.0):  # If the TCE's transit time is within half a bindwidth of the identified bad cadence
                    outfile.write("%12s %14.10f 1\n" % (j['tce'],j['ttime']))    # Write the bad TCE name, transit time, and the number 1 to the output file, to be read by robovetter

    # Close the output file
    outfile.close()



# Run main by default
if __name__ == "__main__":
    main()
