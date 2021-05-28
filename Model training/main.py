import numpy as np
import tensorflow as tf
import keras
from SNNTrainingScript import TrainSNN # Module that gets into the nitty gritty settings to train the neural network according to the values set in "Parameters"

# Parameters
LossLst = ["MAP Error","MS Error"] # MAP Error
ActivationLst = ['sigmoid','relu','tanh'] # sigmoid
ActivationDispLst = ["Sigmoid","Rectified Linear Unit","Hyperbolic Tangent"] # "Sigmoid"
InputPerSectionLst = [64] # 8
OverlappingPixelPercentage = (0,0.5,0.875) # 0
FirstHiddenLayerUnitPercentage = [2] # This number is multiplied with "InputPerSectionLst"
SecondHiddenLayerUnitPercentage = [0] # This number is multiplied with "InputPerSectionLst"


for loss in LossLst:
    for activationCount,activation in enumerate(ActivationLst):
        for inputPerSection in InputPerSectionLst:
            for overlappingRatio in OverlappingPixelPercentage:
                for firstHiddenLayerRatio in FirstHiddenLayerUnitPercentage:
                    for secondHiddenLayerRatio in SecondHiddenLayerUnitPercentage:
                        # Instantiate a dictionary that will store the training parameters for the SNN model
                        trainingParamDict ={
                            # Fixed
                            "Path2i690TrainingData":"", # Path to eLED1's training data
                            "Path2ii690TrainingData":"", # Path to eLED2's training data
                            "Path2SaveWeights":"", # Path to save the trained weights of the different models
                            "NoOfOutput":2,
                            "Epochs":20000,
                            # Varying
                            "Loss":loss,
                            "Activation":activation,
                            "ActivationDisp": ActivationDispLst[activationCount],
                            "InputPerSection":inputPerSection,
                            "OverlappingPixels":int(inputPerSection*overlappingRatio),
                            "FirstLayerHiddenUnits":int(inputPerSection*firstHiddenLayerRatio),
                            "SecondLayerHiddenUnits":int(inputPerSection*secondHiddenLayerRatio)
                            }
                        TrainSNN(trainingParamDict)
