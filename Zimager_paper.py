import cv2
import numpy as np
from PIL import Image
import os
import shutil
import sys
from ximea import xiapi
from time import sleep
import serial
import psychopy.filters
from PyMata.pymata import PyMata
import datetime
import PySpin

#port than the arduino connected to
port="COM3"

#camera to be used 1 = ximea xiC, 2 = ximea xiMU, 3 = FLIR IMX425, 4 = FLIR IMX249
camera = 1

#number of frames in 5Hz frame
framesinframe = 16

#number of frames in a trial after frame binning down from 80Hz to 5Hz
numImagesToGrab = 20

#what are the stims? must be numbers in consecutive order (8bit identifiers for each stim, 0 is always blank)
stims = np.array([0,1,2,3,4,5,6,7,8])

#number of stims (including blank)
numConditions = len(stims)

#number of trials done per stim
trialcount = np.zeros((numConditions), np.uint16)

#number of correct trials done per stim
correctcount = np.zeros((numConditions), np.uint16)

#camera exposure time in microseconds
exposure = 6000

#camera frame rate
FR = 80

#pixel binning amount, if done after acquisition?
binning = 3

#first frame to be used in online map (0=1, 19=20, etc with python)
firstOnlineMapFrame = 8

#last frame to be used in online map
lastOnlineMapFrame = 19

#filter settings
lpkernel = 4
hpkernel = 0.001

#number of online maps, must be 2 or greater
numMaps = 3

#base directory for saving imaging data
root='C:/Users/username/Documents/Data/date/run01/'
os.mkdir(root+'correct')                #this folder is for writing individual tiffs
os.mkdir(root+'incorrect')                #this folder is for writing individual tiffs
os.mkdir(root+'incomplete')                #this folder is for writing individual tiffs

for i in xrange(numConditions):
    os.mkdir(root+'incomplete/'+'condition-%d' % stims[i])                         # create a condition folder for them

#initial values
condition_num = 0
trials = 0
trialmax = 1000

if camera < 3:
    framemultiplier = 65535/float(4095*framesinframe*binning*binning)
else:
    framemultiplier = 1/float(framesinframe*binning*binning)

def generatemaps(Maps,O,X):
    Maps[0] = X[5] + X[6] + X[7] + X[8] - X[1] - X[2] - X[3] - X[4]        #OD
    Maps[1] = X[5] + X[6] + X[7] + X[8] - (4*X[0])                          #right eye
    Maps[2] = X[1] + X[2] + X[3] + X[4] - (4*X[0])                          #left eye
    return Maps

def handshake(port):                          #set up arduino communication
    arduino = PyMata(port, verbose=True)            #for controlling arduino via pymata
    return arduino

def bitread(arduino):
    #insert code for reading input bits via arduino or other input device here
    #print(stimID)
    return stimID

def filter(array,lpkernel,hpkernel):                  #filter data
    kernel = np.ones((lpkernel,lpkernel),np.float64)/(lpkernel*lpkernel)
    array = array - array.mean()
    array = array/array.std()
    array = cv2.filter2D(array,-1,kernel)  
    array_freq = np.fft.fft2(array)
    array_amp = np.fft.fftshift(np.abs(array_freq))
    hp_filt = psychopy.filters.butter2d_hp(size=array.shape,cutoff=hpkernel,n=10)
    array_filt = np.fft.fftshift(array_freq) * hp_filt
    array = np.real(np.fft.ifft2(np.fft.ifftshift(array_filt)))
    array = (array*(65535/(8*array.std()))+32768) # 16 bit clip
    filtered_clipped_array = np.uint16(np.clip(array, a_min=0, a_max=65535)) #16bit conversion
    return filtered_clipped_array


def grabimages(img,framemultiplier,numImagesToGrab,stim,trialID,height,length,W,bining,trial,A,B,B16bit):
    #create arrays to manipulate data in
    #start image acquisition
    while arduino.digital_read(11) == 1:                                       # check for gobit to go off
        sleep(0.0001)                                                             # wait then check again
    #print('Starting data acquisition...')
    if camera < 3:                                              #ximea cameras
        cam.start_acquisition()
        for i in xrange(numImagesToGrab):
            C = W                                               #create variable to put images from camera into
            for j in xrange(framesinframe):
                cam.get_image(img)                              #get data and pass them from camera to img
                data = img.get_image_data_numpy()               #create numpy array with data from camera
                C = C + data                                    #add image to previous images
            A[i] = C[0:(height*binning),0:(length*binning)]    #crops image to standard size (full HD)
    else:
        cam.BeginAcquisition()
        for i in xrange(numImagesToGrab):
            C = W                                               #empty array for image summing
            for j in xrange(framesinframe):
                image = cam.GetNextImage()
                img_conv = image.Convert(PySpin.PixelFormat_Mono16, PySpin.HQ_LINEAR)
                imArray = img_conv.GetNDArray()
                C = C + (imArray)                             #add image to other images summed within this 'frame'
                image.Release()                                 #releasing the image so the buffer doesn't over fill
            A[i] = C
    for i in xrange(numImagesToGrab):
        if binning > 1:
            B[i] = A[i].reshape(height, binning, length, binning).sum(-1,np.float64).sum(1,np.float64)    #pixel binning
        else:
            B[i] = A[i]
        B16bit[i] = np.uint16(B[i]*framemultiplier)                               #scales pixel binned image down to 16bits
        cv2.imwrite((root+'/incomplete'+'/condition-%d' % stim+'/trial-%d' % trial+'-stim-%d' % trialID+'/frame-%d.tiff' % i), B16bit[i])
    return B                    #returns the array of frames from the trial after frame binning and pixel binning

if camera == 1:
    #create variables to hold single condition maps
    V = np.zeros((400,640), np.float64)
    W = np.zeros((1216,1936), np.uint32)
    cam = xiapi.Camera()                        #create instance for first connected camera 
    print('Opening first camera...')
    cam.open_device()                           #start communication
    cam.set_imgdataformat('XI_RAW16')           #set camera image format
    cam.set_exposure(exposure)                  #set camera exposure time
    cam.set_framerate(FR)                       #set camera framerate
    img = xiapi.Image()
    print('connected to ximea IMX174 minicamera')
if camera == 2:
    #create variables to hold single condition maps
    V = np.zeros((486,648), np.float64)
    W = np.zeros((486,648), np.uint32)
    cam = xiapi.Camera()                        #create instance for first connected camera 
    print('Opening first camera...')
    cam.open_device()                           #start communication
    cam.set_imgdataformat('XI_RAW16')           #set camera image format
    cam.set_downsampling('XI_DWN_4x4')          #set binning method
    cam.set_exposure(exposure)                  #set camera exposure time
    cam.set_framerate(FR)                       #set camera framerate
    img = xiapi.Image()
    print('connected to ximea aptina MT9P031 microcamera')
if camera == 3:
    #create variables to hold single condition maps
    V = np.zeros((550,800), np.float64)
    W = np.zeros((550,800), np.uint32)
    serial = '20039881'
    system = PySpin.System.GetInstance()
    camlist = system.GetCameras()
    cam = camlist.GetBySerial(serial)
    cam.Init()
    cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    img = np.zeros((1,550,800), np.uint16) #empty image capture array
    print('connected to FLIR IMX425 camera 20039881')

arduino = handshake(port)
height = len(V[:,1])
length = len(V[1,:])
print('output image size %d ' % length+' x %d' % height)
D = np.zeros((numConditions,height,length), np.float64)
E = np.zeros((numConditions,height,length), np.float64)
D16bit = np.zeros((numConditions,height,length), np.uint16)
E16bit = np.zeros((numConditions,height,length), np.uint16)
Maps = np.zeros((numMaps,height,length), np.float64)
Maps16bit = np.zeros((numMaps,height,length), np.uint16)

while trials < trialmax:
    A = np.zeros((numImagesToGrab,height*binning,length*binning), np.float64)
    B = np.zeros((numImagesToGrab,height,length), np.float64)
    B16bit = np.zeros((numImagesToGrab,height,length), np.uint16)
    print('waiting for stim bits')
    while arduino.digital_read(4) == 0:                                       # waiting for ID bits from behavioral program
        sleep(0.01)
    stimID = bitread(arduino)
    trialID = bitread(arduino)
    trialcount[stimID] = trialcount[stimID] + 1                             # add one to the trial counter for this stim
    trials = trials + 1
    print('condition number: %d' % stimID)
    print('waiting for go signal')
    while arduino.digital_read(11) == 1:                                       # check for gobit to go off
        sleep(0.001)                                                             # wait then check again
    os.mkdir(root+'incomplete/'+'condition-%d' % stimID+'/trial-%d' % trialcount[stimID]+'-stim-%d/' % trialID)
    trialarray = np.int32(grabimages(img,framemultiplier,numImagesToGrab,stimID,trialID,height,length,W,binning,trialcount[stimID],A,B,B16bit))
    print('Stopping acquisition...')
    if camera < 3:
        cam.stop_acquisition()
    else:
        cam.EndAcquisition()
    print(datetime.datetime.utcnow())
    print('ready for trial summary')
    cv2.imwrite((root+'current_redmap.tiff'), np.uint16(trialarray[0]*framemultiplier))
    #arduino.digital_write(3,1)                                              # confirmation signal that collection ended
    while arduino.digital_read(4) == 1:                                       # check for successful trial confirmation
        sleep(0.01)
    outcome = bitread(arduino)                                              # 1 = correct, 2 = incorrect, 0 = aborted
    if outcome == 1:
        for o in xrange(lastOnlineMapFrame-firstOnlineMapFrame):                        # summ frames to create condition maps
            D[stimID] = D[stimID] + trialarray[o+firstOnlineMapFrame] - trialarray[0]
        D16bit[stimID] = np.uint16(np.clip((((D[stimID]-D[stimID].mean())*65535//(8*D[stimID].std()))+32768), a_min=0, a_max=65535))
        cv2.imwrite((root+'/correct_condition-%d.tiff' % stimID), D16bit[stimID])   # write condition maps to file
        correctcount[stimID] = correctcount[stimID] + 1
        shutil.move(root+'/incomplete'+'/condition-%d' % stimID+'/trial-%d' % trialcount[stimID]+'-stim-%d' % trialID+'/', root+'/correct'+'/condition-%d' % stimID+'/trial-%d' % trialcount[stimID]+'-stim-%d' % trialID+'/')
        if  correctcount.max() == correctcount.min():
            Cmaps = generatemaps(Maps,D,E)                                                    # generate online maps
            for p in xrange(numMaps):
                Maps16bit[p] = filter(Cmaps[p],lpkernel,hpkernel)                           # filter online maps
                cv2.imwrite((root+'/subtractionmap-%d.tiff' % p), (Maps16bit[p]))        # write online maps to file
    if outcome == 2:
        for o in xrange(lastOnlineMapFrame-firstOnlineMapFrame):                        # summ frames to create condition maps
            E[stimID] = E[stimID] + trialarray[o+firstOnlineMapFrame] - trialarray[0]
        E16bit[stimID] = np.uint16(np.clip((((E[stimID]-E[stimID].mean())*65535//(8*E[stimID].std()))+32768), a_min=0, a_max=65535))
        cv2.imwrite((root+'/incorrect_condition-%d.tiff' % stimID), E16bit[stimID])   # write condition maps to file
        shutil.move(root+'/incomplete'+'/condition-%d' % stimID+'/trial-%d' % trialcount[stimID]+'-stim-%d' % trialID+'/', root+'/incorrect'+'/condition-%d' % stimID+'/trial-%d' % trialcount[stimID]+'-stim-%d' % trialID+'/')
    sleep(.1)
    arduino.digital_write(3,0)                                          #signal data write completion

#stop communication
if camera < 3:
    cam.close_device()
print 'Done'