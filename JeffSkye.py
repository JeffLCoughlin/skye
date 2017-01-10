# -*- coding: utf-8 -*-

# # Make input files:
# awk '{if(NR>2 && $15>90)  print($1,$15,$187)}' /home/jeff/KeplerJob/DR25/OPS/DATA/TCEs-Full.txt | join - /home/jeff/KeplerJob/DR25/OPS/DATA/TransitTimes.txt | awk 'BEGIN{print("tce  ttime  skygroup  period")} {print($1,$4,$3,$2)}' | sort -k 1n > OPS-TTimes-Skygroup-Per-P90.txt
# awk '{if(NR>2 && $15>90)  print($1,$15,$187)}' /home/jeff/KeplerJob/DR25/INV/DATA/TCEs-Full.txt | join - /home/jeff/KeplerJob/DR25/INV/DATA/TransitTimes.txt | awk 'BEGIN{print("tce  ttime  skygroup  period")} {print($1,$4,$3,$2)}' | sort -k 1n > INV-TTimes-Skygroup-Per-P90.txt
# awk '{if(NR>2 && $15>90)  print($1,$15,$187)}' /home/jeff/KeplerJob/DR25/SS1/DATA/TCEs-Full.txt | join - /home/jeff/KeplerJob/DR25/SS1/DATA/TransitTimes.txt | awk 'BEGIN{print("tce  ttime  skygroup  period")} {print($1,$4,$3,$2)}' | sort -k 1n > SS1-TTimes-Skygroup-Per-P90.txt
#
# awk '{if(NR>2)  print($1,$15,$187)}' /home/jeff/KeplerJob/DR25/OPS/DATA/TCEs-Full.txt | join - /home/jeff/KeplerJob/DR25/OPS/DATA/TransitTimes.txt | awk 'BEGIN{print("tce  ttime  skygroup  period")} {print($1,$4,$3,$2)}' | sort -k 1n > OPS-TTimes-Skygroup-Per.txt
# awk '{if(NR>2)  print($1,$15,$187)}' /home/jeff/KeplerJob/DR25/INV/DATA/TCEs-Full.txt | join - /home/jeff/KeplerJob/DR25/INV/DATA/TransitTimes.txt | awk 'BEGIN{print("tce  ttime  skygroup  period")} {print($1,$4,$3,$2)}' | sort -k 1n > INV-TTimes-Skygroup-Per.txt
# awk '{if(NR>2)  print($1,$15,$187)}' /home/jeff/KeplerJob/DR25/SS1/DATA/TCEs-Full.txt | join - /home/jeff/KeplerJob/DR25/SS1/DATA/TransitTimes.txt | awk 'BEGIN{print("tce  ttime  skygroup  period")} {print($1,$4,$3,$2)}' | sort -k 1n > SS1-TTimes-Skygroup-Per.txt
#
# awk 'BEGIN{print("tce  period  epoch  skygroup")} {if(NR>2) print($1,$15,$18,$187)}' /home/jeff/KeplerJob/DR25/OPS/DATA/TCEs-Full.txt | sort -k 1n > OPS-Per-Epoch-Skygroup.txt
# awk 'BEGIN{print("tce  period  epoch  skygroup")} {if(NR>2) print($1,$15,$18,$187)}' /home/jeff/KeplerJob/DR25/INV/DATA/TCEs-Full.txt | sort -k 1n > INV-Per-Epoch-Skygroup.txt
# awk 'BEGIN{print("tce  period  epoch  skygroup")} {if(NR>2) print($1,$15,$18,$187)}' /home/jeff/KeplerJob/DR25/SS1/DATA/TCEs-Full.txt | sort -k 1n > SS1-Per-Epoch-Skygroup.txt
# awk 'BEGIN{print("tce  period  epoch  skygroup")} {if(NR>2) print($1,$15,$18,$187)}' /home/jeff/KeplerJob/DR25/INJ/DATA/TCEs-Full.txt | sort -k 1n > INJ-Per-Epoch-Skygroup.txt
#
# Run code:
# python -c "execfile('/home/jeff/KeplerJob/DR25/Code/JeffSkye.py'); calcSkye(1.0,4.5,'OPS-TTimes-Skygroup-Per-P90.txt','skye-ops.txt','OPS-skye-plots.pdf')"
# python -c "execfile('/home/jeff/KeplerJob/DR25/Code/JeffSkye.py'); calcSkye(1.0,4.5,'INV-TTimes-Skygroup-Per-P90.txt','skye-inv.txt','INV-skye-plots.pdf')"
# python -c "execfile('/home/jeff/KeplerJob/DR25/Code/JeffSkye.py'); calcSkye(1.0,4.5,'SS1-TTimes-Skygroup-Per-P90.txt','skye-ss1.txt','SS1-skye-plots.pdf')"
# #
# python -c "execfile('/home/jeff/KeplerJob/DR25/Code/JeffSkye.py'); calcBadTCETransits(1.0,'OPS-Per-Epoch-Skygroup.txt','skye-ops.txt','Skyline-Metric-OPS-jeff.txt')"
# python -c "execfile('/home/jeff/KeplerJob/DR25/Code/JeffSkye.py'); calcBadTCETransits(1.0,'INV-Per-Epoch-Skygroup.txt','skye-inv.txt','Skyline-Metric-INV-jeff.txt')"
# python -c "execfile('/home/jeff/KeplerJob/DR25/Code/JeffSkye.py'); calcBadTCETransits(1.0,'SS1-Per-Epoch-Skygroup.txt','skye-ss1.txt','Skyline-Metric-SS1-jeff.txt')"
# python -c "execfile('/home/jeff/KeplerJob/DR25/Code/JeffSkye.py'); calcBadTCETransits(1.0,'INJ-Per-Epoch-Skygroup.txt','skye-ops.txt','Skyline-Metric-INJ-jeff.txt')"


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


# This function calculates and returns the MAD of the input values, the MAD of the values above the median, and the MAD of the values below the median
def DoubleMAD(x):
   m      = np.median(x)
   absdev = np.abs(x - m)
   negmad = np.median(absdev[x<=m])
   posmad = np.median(absdev[x>=m])
   midmad = np.median(absdev)
   return 1.4826*midmad, 1.4826*posmad, 1.4826*negmad  # Return corrected value to equate to stdev


# Calculate which transits of which TCEs correspond to bad cadences on particular skygroups
def calcBadTCETransits(binwidth,filenamein,skyefilename,outfilename):
    """Compute times of bad transits using Skye results

    Inputs:
    -------------
    binwidth
        The bin width, in days, for Skye

    filenamein
        The name of the file containing the tces, periods, epochs, and skygroups

    skyefilename
        The output file from calcSkye


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
    for i in skyedat:   # Loop through each bad transit time / skygroup pair identified by Skye
        for j in data:  # Loop through each TCE
            if(i[1] == j['skygroup']):  # If the TCE is on the skygroup (saves a lot of comp time to check first here)
                t = j['epoch']  # Start computing the time of transit starting from the given epoch
                while(t < maxt):    # Loop through all possibe transit times given the TCE's ephemeris and maximum time of the data set
                    if(t > i[0]-binwidth/2 and t < i[0]+binwidth/2):  # If the TCE's transit time is within half a bindwidth of the identified bad cadence
                        outfile.write("%12s %14.10f 1\n" % (j['tce'],t))    # Write the bad TCE name, transit time, and the number 1 to the output file, to be read by robovetter
                    t += j['period']    # Check the TCE's next transit time

    # Close the output file
    outfile.close()




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

    ttimes = data['ttime'][(data['period']>90)]     # Select TCEs that only have periods great than 90 days
    # skygroups = data['skygroup'][(data['period']>90)]
    # periods = data['period'][(data['period']>90)]

    mint = 131.5
    maxt = 1591.0

    hist = np.histogram(ttimes,bins=np.arange(mint, maxt + binwidth, binwidth))
    histvals = hist[0]
    bins = hist[1]
    bincenters = 0.5*(bins[1:]+bins[:-1])   # Compute the center time of each bin

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
        ttimes = data[np.logical_and(data['skygroup']==sg,data['period']>90)]['ttime']

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
