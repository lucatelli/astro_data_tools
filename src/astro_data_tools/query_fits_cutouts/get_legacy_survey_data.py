#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""

Date = '09/2025'
Geferson Lucatelli

Usage:
$ python3 /path/to/catalogue.csv /path/to/output_dir/
Your catalogue.csv must have #ID,RA,DEC.

  _                                     ____
| |    ___  __ _  __ _  ___ _   _     / ___| _   _ _ ____   _____ _   _
| |   / _ \/ _` |/ _` |/ __| | | |    \___ \| | | | '__\ \ / / _ \ | | |
| |__|  __/ (_| | (_| | (__| |_| |     ___) | |_| | |   \ V /  __/ |_| |
|_____\___|\__, |\__,_|\___|\__, |    |____/ \__,_|_|    \_/ \___|\__, |
           |___/            |___/                                 |___/
                  ____      _              _       
                 / ___|   _| |_ ___  _   _| |_ ___ 
                | |  | | | | __/ _ \| | | | __/ __|
                | |__| |_| | || (_) | |_| | |_\__ \
                 \____\__,_|\__\___/ \__,_|\__|___/

"""
from __future__ import division
import numpy as np
import pylab as pl
import astropy.io.fits as pf
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")
from progress.bar import Bar
import requests
import multiprocessing
from tqdm import tqdm
from joblib import Parallel, delayed
from sys import argv



def grab_images():
    def get_data(File,param=None,HEADER=0):
        """
        Get a numerical variable from a table.

        HEADER: if ==1, display the file's header.
        """
        infile = open(File, 'r')
        firstLine = infile.readline()
        header=firstLine.split(',')
        if HEADER==1:
            print(header)
            return(header)
        else:
            ind=header.index(param)
            return np.loadtxt(File,usecols=(ind),comments="#", delimiter=",", \
                unpack=False)
    def getstr(File,string=None,HEADER=0):
        """
        Get a string variable from a table.
        May work for float as well.
        """
        infile = open(File, 'r')
        firstLine = infile.readline()
        header=firstLine.split(',')
        if HEADER==1:
            print(header)
            return(header)
        else:
            ind=header.index(string)
            return np.loadtxt(File,dtype='str',usecols=(ind),comments="#", \
            delimiter=",", unpack=False)

    file = str(argv[1])
    _file_name = os.path.splitext(os.path.basename(file))
    file_name = _file_name[0]+_file_name[1]
    #where to save the stamps
    # save_path = "path_to_where_to_save_the_stamps/"+file_name.replace(".csv","")+"/"
    # save_path = str(argv[2])+_file_name[0].replace(".csv","")+"/"
    save_path = str(argv[2])+_file_name[0]+"/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    try:
        ra     = get_data(File=file,param='RA')
    except:
        ra     = get_data(File=file,param='#RA')

    dec     = get_data(File=file,param='DEC')

    try:
        IDs = getstr(File=file,string='ID')
    except:
        try:
            IDs = getstr(File=file,string='#ID')
        except:
            # IDs = np.arange(1,len(ra)+1).astype(str)
            IDs = []
            for i in range(len(ra)):
                radec = ra[i].astype(str) +'_'+ dec[i].astype(str)
                IDs.append(radec)
            IDs=np.asarray(IDs)

    try:
        # ai = get_data(File=file,param='A')
        # bi = get_data(File=file,param='B')
        # Radius = 20*np.sqrt(ai**2.0+bi**2.0)
        D25 = get_data(File=file,param='D25')
        Radius = ((D25 * 360)/2.0) / 0.26 #in pixels, for HSC pixel scale.
        for k in range(len(Radius)):
            if D25[k]<0:
                Radius[k]=int(128*3)
    except:
        size = int(128*3)
        Radius = np.ones(len(IDs))
        Radius =Radius*size


    BAND = ["r"]#,"g","z"]#for all bands in a same fits file, set 'grz' altogether.


    def get_images_ra_dec(ID,ra,dec,Radius,band="r",path_to_save = ""):
        root_url = "https://www.legacysurvey.org/viewer/cutout.fits?"
        # root_url = "https://www.legacysurvey.org/viewer/cutout.fits?ra=359.9848&dec=0.6751&layer=ls-dr8&pixscale=0.263"
        # ra=359.9848&dec=0.6751&layer=ls-dr8&pixscale=0.263
        # root_url = "https://datalab.noao.edu/svc/cutout?col=splus_dr1&siaRef="
        survey = "ls-dr9"#"hsc-dr2"#"decals-dr7"#"ls-dr9"#"sdss"

        # 0.17 for subaru (hsc-dr2), 0.27 for ls-dr9, etc. 
        pixscale=0.26#0.17
        size=Radius
        
        url = root_url + "ra="+str(ra)+"&dec="+str(dec)+"&size="+str(size)\
            +"&layer="+str(survey)+"&pixscale="+str(pixscale)+"&bands="+str(band)
        try:
            if not os.path.exists(save_path+band+"/"+ID+"_"+band+".fits"):
                r = requests.get(url,verify=True,timeout=10)
                # with open(path_to_save+"SPLUS."+STRIPE+"-"+ID+".griz_"+band+".fits",'wb') as f:
                with open(path_to_save+band+"/"+ID+"_"+band+".fits",'wb') as f:
                    f.write(r.content)
                f.close()
        except:
            try:
                if not os.path.exists(save_path+band+"/"+ID+"_"+band+".fits"):
                    r = requests.get(url,verify=True,timeout=10)
                    # with open(path_to_save+"SPLUS."+STRIPE+"-"+ID+".griz_"+band+".fits",'wb') as f:
                    with open(path_to_save+band+"/"+ID+"_"+band+".fits",'wb') as f:
                        f.write(r.content)
                    f.close()
            except:
                print('---------------------------------------')
                print('requests.get timeout error.')
                print('Skipping ID=',ID)
                print('         ra,dec=',ra,dec)
                print('Trying again later')
                print('---------------------------------------')

    for band in BAND:
        print("Downloading images for the",band,' band.')
        if not os.path.exists(save_path+band):
            os.makedirs(save_path+band)

        """
        Originally, setting a high value of NProc (number of parallel downloads) was stable.
        For example, to download ~18000 images (256x256), it takes about 30minutes.
        However, that is not working anymore in 2025. 
        """
        NProc = 4 #it is no longer possible to use high values than 4~6. 
        processed_list = Parallel(n_jobs=NProc)(\
                         delayed(get_images_ra_dec)(\
                            IDs[k],ra[k],dec[k],int(Radius[k]),band,\
                            path_to_save=save_path) for k in tqdm(range(len(IDs))))

        def try_missing():
            """
            If any error occured during the first try, this function will check for
            those missing images and will try to download them again.
            """
            print('----------------------')
            print('Checking missing files')
            print('----------------------')
            IDs_Re = []
            ra_Re = []
            dec_Re = []
            Radius_Re = []
            for k in range(len(IDs)):
                if not os.path.exists(save_path+band+"/"+IDs[k]+"_"+band+".fits"):
                    IDs_Re.append(IDs[k])
                    ra_Re.append(ra[k])
                    dec_Re.append(dec[k])
                    Radius_Re.append(Radius[k])
                else:
                    pass
            if len(IDs_Re)>0:
                print('-----------------------------------------------------')
                print(len(IDs_Re),'missing files. Trying to grab them again.')
                print('-----------------------------------------------------')
                NProc = 3 #small value to avoid connections issues.
                processed_list = Parallel(n_jobs=NProc)(\
                                delayed(get_images_ra_dec)(\
                                    IDs_Re[k],ra_Re[k],dec_Re[k],int(Radius_Re[k]),band,\
                                    path_to_save=save_path) for k in tqdm(range(len(IDs_Re))))
                print('Done.')
            else:
                print('-----------------')
                print('No missing files.')
                print('Done.')
                print('-----------------')
        try_missing()

if __name__ == '__main__':
    grab_images()
