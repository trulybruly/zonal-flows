import urllib.request
import requests
import os
import numpy as np

directory = './golemcache/'
target_files = ['DRP-L1.csv', 'DRP-R1.csv',
                'DRP-L2.csv', 'DRP-R2.csv', 
                'DRP-L3.csv', 'DRP-R3.csv',
                'DRP-L4.csv', 'DRP-R4.csv',
                'DRP-L5.csv', 'DRP-R5.csv',
                'DRP-L6.csv', 'DRP-R6.csv',
                'CameraRadialPosition.csv', 'CameraVerticalPosition.csv',
                'U_IntBtCoil.csv', 'U_Loop.csv', 'U_IntRogCoil.csv',
                'ne_lav.csv', '	ne_lav_unrepaired.csv',
                'ring_1.csv', 'ring_2.csv', 'ring_3.csv', 'ring_4.csv', 'ring_5.csv', 'ring_6.csv',
                'ring_7.csv', 'ring_8.csv', 'ring_9.csv', 'ring_10.csv', 'ring_11.csv', 'ring_12.csv', 
                'ring_13.csv', 'ring_14.csv', 'ring_15.csv', 'ring_16.csv',
                'U_mc1.csv', 'U_mc5.csv', 'U_mc9.csv', 'U_mc13.csv',
                'U_BPP-bottom.csv', 'U_BPP-upper.csv', 'U_LP-bottom.csv', 'U_LP-upper.csv',
                ]

def folderscan(shot, curdir, folders, rewrite):
    yes = ['Y', 'y', 1, '1', True]
    no = ['N', 'n', 0, '0', False] 
        
    response = requests.get('http://golem.fjfi.cvut.cz/shotdir/{}'.format(curdir))
    curhtml = response.text
    curhtml = curhtml.split('\n')[11:-5]
    files = []

    for i in curhtml:
        if 'folder' in i:
            j = i.split('href="')[1]
            k = j.split('"')[0]
            if not ((len(k) == 7) and k[-1] == '/'):
                folders.append(curdir+'/'+k)
        else:
            j = i.split('href="')[1]
            k = j.split('"')[0]
            if any(target_file in k for target_file in target_files):
                files.append(k) 
    
    if rewrite in yes:
        for f in files:
            file_name = curdir+'/'+f
            f1 = shot + '/' + f
            url = 'http://golem.fjfi.cvut.cz/shotdir/{}'.format(file_name)
            print(url)
            if not os.path.exists(directory + file_name):
                urllib.request.urlretrieve(url, directory + f1)
                print('Downloaded ' + directory + f1)
            else:
                os.remove(directory + file_name)
                urllib.request.urlretrieve(url, directory + f1)
                print('Reloaded ' + directory + f1)

    elif rewrite in no:
        for f in files:
            file_name = curdir+'/'+f
            f1 = shot + '/' + f
            url = 'http://golem.fjfi.cvut.cz/shotdir/{}'.format(file_name)
            print('Trying ' + url)
            if not os.path.exists(directory + file_name):
                urllib.request.urlretrieve(url, directory + f1)
                print('Downloaded ' + directory + f1)
            else:
                print('Passed ' + directory + f1)

def downloadshot(shotnum, rewrite=True):
    shotnum = str(shotnum)
    if not os.path.exists(directory):
        os.mkdir(directory)
    if not os.path.exists(directory+shotnum):
        os.mkdir(directory+shotnum)   
    folders = []
    folderscan(shotnum, shotnum, folders, rewrite)
    
    while folders != []:
        curdir = folders.pop(0)
        folderscan(shotnum, curdir, folders, rewrite)

shotlist = np.arange(49019, 49020, 1)

for shot in shotlist:
    downloadshot(shot, True)