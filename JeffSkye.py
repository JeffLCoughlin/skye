# -*- coding: utf-8 -*-

# Make input files:
# awk '{if(NR>2 && $15>90)  print($1,$15,$187)}' /home/jeff/KeplerJob/DR25/OPS/DATA/TCEs-Full.txt | join - /home/jeff/KeplerJob/DR25/OPS/DATA/TransitTimes.txt | awk 'BEGIN{print("ttime  skygroup  period")} {print($4,$3,$2)}' | sort -k 1n > OPS-TTimes-Skygroup-Per-P90.txt
# awk '{if(NR>2 && $15>90)  print($1,$15,$187)}' /home/jeff/KeplerJob/DR25/INV/DATA/TCEs-Full.txt | join - /home/jeff/KeplerJob/DR25/INV/DATA/TransitTimes.txt | awk 'BEGIN{print("ttime  skygroup  period")} {print($4,$3,$2)}' | sort -k 1n > INV-TTimes-Skygroup-Per-P90.txt
# awk '{if(NR>2 && $15>90)  print($1,$15,$187)}' /home/jeff/KeplerJob/DR25/SS1/DATA/TCEs-Full.txt | join - /home/jeff/KeplerJob/DR25/SS1/DATA/TransitTimes.txt | awk 'BEGIN{print("ttime  skygroup  period")} {print($4,$3,$2)}' | sort -k 1n > SS1-TTimes-Skygroup-Per-P90.txt
#
# Run code:
# python -c "execfile('/home/jeff/KeplerJob/DR25/Code/JeffSkye.py'); calcSkye(1.0,4.5,'OPS-TTimes-Skygroup-Per-P90.txt','skye-ops.txt','OPS-skye-plots.pdf')"
# python -c "execfile('/home/jeff/KeplerJob/DR25/Code/JeffSkye.py'); calcSkye(1.0,4.5,'INV-TTimes-Skygroup-Per-P90.txt','skye-inv.txt','INV-skye-plots.pdf')"
# python -c "execfile('/home/jeff/KeplerJob/DR25/Code/JeffSkye.py'); calcSkye(1.0,4.5,'SS1-TTimes-Skygroup-Per-P90.txt','skye-ss1.txt','SS1-skye-plots.pdf')"



from subprocess import check_output, CalledProcessError
import signal
import numpy as np
import os
import scipy
import math
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import special
import sys
import matplotlib.colors as colors
from astropy.stats import biweight_midvariance
from statsmodels.robust.scale import huber
import peakutils

reload(sys)
sys.setdefaultencoding('utf8')


def DoubleMAD(x):
   m      = np.median(x)
   absdev = np.abs(x - m)
   negmad = np.median(absdev[x<=m])
   posmad = np.median(absdev[x>=m])
   midmad = np.median(absdev)
   return 1.4826*midmad, 1.4826*posmad, 1.4826*negmad  # Return corrected value to equate to stdev



def calcSkye(binwidth,sigma,filenamein,filenameout,plotname,opt=None):
    """Compute times of transit clusters for each skygroup

    Inputs:
    -------------
    binwidth
        The bin width, in days, for Skye

    filename
        The name of the file containing the times of transit, skygroup, and period

    Returns:
    -------------
    None

    Output:
    ----------
    File containing times and skygroups, upon which any transits should be considered 'bad'
    Plot showing histogram distribution and cutoff value
    """

    pdf_pages = PdfPages(plotname)
    outfile = open(filenameout, 'w')

    # Load from RoboVetter-Input-INJ-PlanetOn-ForErrors.txt
    data = np.genfromtxt(filenamein, names=True, dtype=None)

    ttimes = data['ttime']
    skygroups = data['skygroup']
    periods = data['period']
    # mint = min(ttimes)
    # maxt = max(ttimes)
    mint = 131.5
    maxt = 1591.0

    # print(mint,maxt)

    hist = np.histogram(ttimes,bins=np.arange(mint, maxt + binwidth, binwidth))
    histvals = hist[0]
    bins = hist[1]
    bincenters = 0.5*(bins[1:]+bins[:-1])

    # print(np.sum(histvals))

    # histmed = np.median(histvals)
    # midmad, posmad, negmad = DoubleMAD(histvals)
    # cutoff = histmed + sigma*posmad
    # print(histmed,midmad,posmad,cutoff)

    # histmean = np.mean(histvals)
    # stdev = np.std(histvals)
    # cutoff = histmean + sigma*stdev
    # print(histmean,stdev,cutoff)
    # print(len(bincenters))
    # badtimes = bincenters[histvals>cutoff]
    #
    # for time in badtimes:
    #     for sg in range(1,85):
    #         outfile.write("%6.2f %2i\n" % (time,sg))

    # from scipy.signal import find_peaks_cwt
    # indexes = find_peaks_cwt(histvals, np.arange(1,2), min_snr=4)
    # print(bincenters[indexes])
    # print("---")
    # indexes = peakutils.indexes(histvals, thres=0.3, min_dist=0)
    # print(bincenters[indexes])
    # print(indexes)
    # print(histvals[indexes])
    # print(histvals[indexes]>2)
    # print(indexes[(histvals[indexes]>2)])
    # print(bincenters[indexes[(histvals[indexes]>2)]])


    histvalstmp = histvals
    histmean = np.mean(histvalstmp)
    stdev = np.std(histvals)
    cutoff = histmean + sigma*stdev
    # print(histmean,stdev,cutoff)
    for i in range(0,5):
        histvalstmp = histvalstmp[([histvalstmp<cutoff])]
        histmean = np.mean(histvalstmp)
        stdev = np.std(histvalstmp)
        cutoff = histmean + sigma*stdev
        # print(histmean,stdev,cutoff)
    badtimes = bincenters[histvals>cutoff]

    # for time in badtimes:
    #     for sg in range(1,85):
    #         outfile.write("%6.2f %2i\n" % (time,sg))

    print(" ")

    # Make plot for all skygroups
    fig = plt.figure()
    fig.set_size_inches(11,8.5)
    plt.title("All Skygroups\n")
    plt.xlabel("Transit Time")
    plt.ylabel("Number")
    plt.axhline(y=cutoff, c="blue")
    plt.step(bincenters,histvals, color='k', rasterized=True)
    pdf_pages.savefig(fig, dpi=400)

    # pdf_pages.close()
    # exit(0)

    #for sg in range(1,85):
    for sg in range(1,85):
        ttimes = data[(data['skygroup']==sg)]['ttime']

        hist = np.histogram(ttimes,bins=np.arange(mint, maxt + binwidth, binwidth))
        histvals = hist[0]
        bins = hist[1]
        bincenters = 0.5*(bins[1:]+bins[:-1])

        print("Skygroup ",sg)
        # print(np.sum(histvals))
        # histmed = np.median(histvals)
        # midmad, posmad, negmad = DoubleMAD(histvals)
        # cutoff = histmed + sigma*posmad
        # print(histmed,midmad,posmad,cutoff)

        histvalstmp = histvals
        histmean = np.mean(histvalstmp)
        stdev = np.std(histvals)
        cutoff = histmean + sigma*stdev
        # print(histmean,stdev,cutoff)
        for i in range(0,5):
            histvalstmp = histvalstmp[([histvalstmp<cutoff])]
            histmean = np.mean(histvalstmp)
            stdev = np.std(histvalstmp)
            cutoff = histmean + sigma*stdev
            # print(histmean,stdev,cutoff)

        badtimes = bincenters[histvals>cutoff]

        # indexes = find_peaks_cwt(histvals, np.arange(1,2), min_snr=4)
        # print(bincenters[indexes])
        # print("---")
        # indexes = peakutils.indexes(histvals, thres=0.3, min_dist=0)
        # print(bincenters[indexes])
        # print(indexes)
        # print(histvals[indexes])
        # print(histvals[indexes]>2)
        # print(indexes[(histvals[indexes]>2)])
        # print(bincenters[indexes[(histvals[indexes]>2)]])

        for time in badtimes:
            outfile.write("%6.2f %2i\n" % (time,sg))

        print(" ")

        fig = plt.figure()
        fig.set_size_inches(11,8.5)
        plt.title("Skygroup %i\n" % (sg))
        plt.xlabel("Transit Time")
        plt.ylabel("Number")
        plt.axhline(y=cutoff, c="blue")
        plt.step(bincenters,histvals, color='k', rasterized=True)
        pdf_pages.savefig(fig, dpi=400)


    outfile.close()
    pdf_pages.close()
