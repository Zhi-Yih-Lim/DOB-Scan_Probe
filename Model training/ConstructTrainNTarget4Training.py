import numpy as np
from ConvertVolumetoOpticalProperties import Volume2OpticalPpt

def TrainNTargetValues(SectionNumber,PxlShift,InputPerSection,Path2TrainingCsv):
    # Extract the contents of the Training CSV
    data = np.genfromtxt(Path2TrainingCsv,delimiter=",")
    # Compute the pixel range from which to extract the reflectance values 
    pxlRange = [i+2 for i in range ((SectionNumber*PxlShift),(SectionNumber*PxlShift)+InputPerSection)]
    # Based on the computed input range, extract the reflectance values from the csv file, noting that the first two columns contains 
    # the values for the Intralipid and Ink Volumes
    IntNInkVol = data[:,0:2].reshape((data.shape[0],2))
    # Extract the reflectance values
    RescaledNNormRefl = data[:,pxlRange].reshape((data.shape[0],InputPerSection))
    # Calculate the optical properties from the Intralipid and Ink Volumes
    OptPpt = Volume2OpticalPpt(IntNInkVol)
    # Return the Target and Training Values
    return RescaledNNormRefl,OptPpt
