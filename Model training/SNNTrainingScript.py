import numpy as np
import os 
from ConstructTrainNTarget4Training import TrainNTargetValues # Module that divides the training data from .csv format to input and target values
from SNN import SingleLayerNeuralNet # Module that returns a "model" object from keras, to be used for model training (single layered neural network in this case)
from SNNTwoLayer import DoubleLayerNeuralNet # Module that returns a "model" object from keras, to be used for model training (double layered neural network in this case)
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import backend as K

def TrainSNN(TrainingParam):
    # Calculate the pixel shift
    pxlShift = TrainingParam["InputPerSection"]-TrainingParam["OverlappingPixels"]
    # Calculate the total number of sections 
    ttlSection = int((128-TrainingParam["OverlappingPixels"])//pxlShift)
    # Instantiate an array to store the minimum validation error for each section, for the current model input settings
    sectErr = np.zeros((1,ttlSection))
    # Train each section 
    for section in range(ttlSection):
        if section <= int(ttlSection//2): ################################ Change here for section numbers corresponding to different LEDs #################################################
            # Construct the input and target values for the current section
            i690Train,i690Target = TrainNTargetValues(section,pxlShift,TrainingParam["InputPerSection"],TrainingParam["Path2i690TrainingData"])
            # Split the data into training and validation
            X_train,X_valid,Y_train,Y_valid = train_test_split(i690Train,i690Target,test_size=0.05,random_state=3)
            # Construct file path to folder specifying the type of loss function used when training
            pat2TypeOfLossFunc = TrainingParam["Path2SaveWeights"] + "\\{}".format(TrainingParam["Loss"])
            # Create the folder if it has not been previously created
            if not os.path.exists(pat2TypeOfLossFunc):
                os.makedirs(pat2TypeOfLossFunc)
            # Construct file path to folder specifying the type of activation used for training
            pat2ActivationType = pat2TypeOfLossFunc + "\\{}".format(TrainingParam["ActivationDisp"])
             # Create the folder if it has not been previously created
            if not os.path.exists(pat2ActivationType):
                os.makedirs(pat2ActivationType)
            # Construct the file path to the folder of the current number of pixels per section
            pat2NumberOfInputPxl = pat2ActivationType + "\\{} Input".format(TrainingParam["InputPerSection"])
            # Create the folder if it has not been previously created
            if not os.path.exists(pat2NumberOfInputPxl):
                os.makedirs(pat2NumberOfInputPxl)
            # Construct the file path to the folder of the current number of overlapping pixels in between sections
            pat2NumberOfOverlap = pat2NumberOfInputPxl + "\\{} Overlap".format(TrainingParam["OverlappingPixels"])
            # Create the folder if it has not been previously created
            if not os.path.exists(pat2NumberOfOverlap):
                os.makedirs(pat2NumberOfOverlap)
            # Create a temporary SNN model for training 
            if TrainingParam["SecondLayerHiddenUnits"] == 0:
                # Path to folder for different layers of hidden units
                pat2LayerOfHiddenUnits = pat2NumberOfOverlap + "\\OneHidden"
                # Create the folder if it has not been previously created
                if not os.path.exists(pat2LayerOfHiddenUnits):
                    os.makedirs(pat2LayerOfHiddenUnits)
                # Path to folder containing the number of hidden units
                pat2NumberOfHiddenUnits = pat2LayerOfHiddenUnits + "\\{} Hidden First".format(TrainingParam["FirstLayerHiddenUnits"])
                # Create the folder if it has not been previously created
                if not os.path.exists(pat2NumberOfHiddenUnits):
                    os.makedirs(pat2NumberOfHiddenUnits)
                # At the weight destination file path, create a folder for the current section to store its weight file 
                pat2SectionFold = pat2NumberOfHiddenUnits + "\\Section {}".format(section+1)
                # Create the folder if it has not been previously created
                if not os.path.exists(pat2SectionFold):
                    os.makedirs(pat2SectionFold)
                # For each section, record the file from which the training data is obtained and the columns indeces used for training the data
                f = open(pat2SectionFold + "\\SectionInformation.txt", "a")
                f.write("Currently in section {}, Csv file used for training is {}, The indeces used for training are {}".format(section,TrainingParam["Path2i690TrainingData"],[i+2 for i in range ((section*pxlShift),(section*pxlShift)+TrainingParam["InputPerSection"])]))
                f.close()
                # Construct the path to save the weight file 
                pat2Weight = pat2SectionFold + "\\i690{}Section{}.h5".format(section+1,section+1)
                 # Based on the selected metrics, instantaite a new metrics class for every section
                if TrainingParam["Loss"] == "MAP Error":
                    loss = keras.losses.MeanAbsolutePercentageError()
                    metrics = keras.metrics.MeanAbsolutePercentageError()
                    callbackMonitorQuality = "val_mean_absolute_percentage_error"
                elif TrainingParam["Loss"] == "MS Error":
                    loss = keras.losses.MeanSquaredError()
                    metrics = keras.metrics.MeanSquaredError()
                    callbackMonitorQuality = "val_mean_squared_error"
                # Checkpoint to save the best weights
                checkpoint_callback = keras.callbacks.ModelCheckpoint(
                    filepath=pat2Weight,
                    save_weights_only=True,
                    verbose = 1,
                    monitor=callbackMonitorQuality,
                    mode='min',
                    save_best_only=True)
                # Create the model
                tempMod = SingleLayerNeuralNet(InputsPerSection=TrainingParam["InputPerSection"],HiddenUnits=TrainingParam["FirstLayerHiddenUnits"],Activation=TrainingParam["Activation"],Optimizer=keras.optimizers.Adam(lr=0.01),Loss=loss,Metrics=metrics,NoOutput=TrainingParam["NoOfOutput"])
            else:
                # Path to folder for different layers of hidden units
                pat2LayerOfHiddenUnits = pat2NumberOfOverlap + "\\TwoHidden"
                # Create the folder if it has not been previously created
                if not os.path.exists(pat2LayerOfHiddenUnits):
                    os.makedirs(pat2LayerOfHiddenUnits)
                # Path to folder containing the number of hidden units
                pat2NumberOfHiddenUnits = pat2LayerOfHiddenUnits + "\\{} First {} Second".format(TrainingParam["FirstLayerHiddenUnits"],TrainingParam["SecondLayerHiddenUnits"])
                # Create the folder if it has not been previously created
                if not os.path.exists(pat2NumberOfHiddenUnits):
                    os.makedirs(pat2NumberOfHiddenUnits)
                # At the weight destination file path, create a folder for the current section to store its weight file 
                pat2SectionFold = pat2NumberOfHiddenUnits + "\\Section {}".format(section+1)
                # Create the folder if it has not been previously created
                if not os.path.exists(pat2SectionFold):
                    os.makedirs(pat2SectionFold)
                # For each section, record the file from which the training data is obtained and the columns indeces used for training the data
                f = open(pat2SectionFold + "\\SectionInformation.txt", "a")
                f.write("Currently in section {}, Csv file used for training is {}, The indeces used for training are {}".format(section,TrainingParam["Path2i690TrainingData"],[i+2 for i in range ((section*pxlShift),(section*pxlShift)+TrainingParam["InputPerSection"])]))
                f.close()
                # Construct the path to save the weight file 
                pat2Weight = pat2SectionFold + "\\i690{}Section{}.h5".format(section+1,section+1)
                # Based on the selected metrics, instantaite a new metrics class for every section
                if TrainingParam["Loss"] == "MAP Error":
                    loss = keras.losses.MeanAbsolutePercentageError()
                    metrics = keras.metrics.MeanAbsolutePercentageError()
                    callbackMonitorQuality = "val_mean_absolute_percentage_error"
                elif TrainingParam["Loss"] == "MS Error":
                    loss = keras.losses.MeanSquaredError()
                    metrics = keras.metrics.MeanSquaredError()
                    callbackMonitorQuality = "val_mean_squared_error"
                # Checkpoint to save the best weights
                checkpoint_callback = keras.callbacks.ModelCheckpoint(
                    filepath=pat2Weight,
                    save_weights_only=True,
                    verbose = 1,
                    monitor=callbackMonitorQuality,
                    mode='min',
                    save_best_only=True)
                # Create the model
                tempMod = DoubleLayerNeuralNet(InputsPerSection=TrainingParam["InputPerSection"],FirstLayerHiddenUnits=TrainingParam["FirstLayerHiddenUnits"],SecondLayerHiddenUnits=TrainingParam["SecondLayerHiddenUnits"],Activation=TrainingParam["Activation"],Optimizer=keras.optimizers.Adam(lr=0.01),Loss=loss,Metrics=metrics,NoOutput=TrainingParam["NoOfOutput"])
            # Train the model
            history = tempMod.model.fit(X_train,Y_train,batch_size=int(X_train.shape[0]//2),epochs=TrainingParam["Epochs"],validation_data=(X_valid,Y_valid),callbacks=[checkpoint_callback])
            # Save the minimum validation error for the current section
            sectErr[0,section] = min(history.history[callbackMonitorQuality])
            # Clear the training session for the current section
            K.clear_session()# Checked
        else:
            # Compute the section number for LED 2
            LED2Section = ttlSection - section - 1
            # Construct the input and target values for the current section
            ii690Train,ii690Target = TrainNTargetValues(LED2Section,pxlShift,TrainingParam["InputPerSection"],TrainingParam["Path2ii690TrainingData"])
            # Split the data into training and validation
            X_train,X_valid,Y_train,Y_valid = train_test_split(ii690Train,ii690Target,test_size=0.05,random_state=3)
            # Construct file path to folder specifying the type of loss function used when training
            pat2TypeOfLossFunc = TrainingParam["Path2SaveWeights"] + "\\{}".format(TrainingParam["Loss"])
            # Create the folder if it has not been previously created
            if not os.path.exists(pat2TypeOfLossFunc):
                os.makedirs(pat2TypeOfLossFunc)
            # Construct file path to folder specifying the type of activation used for training
            pat2ActivationType = pat2TypeOfLossFunc + "\\{}".format(TrainingParam["ActivationDisp"])
             # Create the folder if it has not been previously created
            if not os.path.exists(pat2ActivationType):
                os.makedirs(pat2ActivationType)
            # Construct the file path to the folder of the current number of pixels per section
            pat2NumberOfInputPxl = pat2ActivationType + "\\{} Input".format(TrainingParam["InputPerSection"])
            # Create the folder if it has not been previously created
            if not os.path.exists(pat2NumberOfInputPxl):
                os.makedirs(pat2NumberOfInputPxl)
            # Construct the file path to the folder of the current number of overlapping pixels in between sections
            pat2NumberOfOverlap = pat2NumberOfInputPxl + "\\{} Overlap".format(TrainingParam["OverlappingPixels"])
            # Create the folder if it has not been previously created
            if not os.path.exists(pat2NumberOfOverlap):
                os.makedirs(pat2NumberOfOverlap)
            # Create a temporary SNN model for training 
            if TrainingParam["SecondLayerHiddenUnits"] == 0:
                # Path to folder for different layers of hidden units
                pat2LayerOfHiddenUnits = pat2NumberOfOverlap + "\\OneHidden"
                # Create the folder if it has not been previously created
                if not os.path.exists(pat2LayerOfHiddenUnits):
                    os.makedirs(pat2LayerOfHiddenUnits)
                # Path to folder containing the number of hidden units
                pat2NumberOfHiddenUnits = pat2LayerOfHiddenUnits + "\\{} Hidden First".format(TrainingParam["FirstLayerHiddenUnits"])
                # Create the folder if it has not been previously created
                if not os.path.exists(pat2NumberOfHiddenUnits):
                    os.makedirs(pat2NumberOfHiddenUnits)
                # At the weight destination file path, create a folder for the current section to store its weight file 
                pat2SectionFold = pat2NumberOfHiddenUnits + "\\Section {}".format(section+1)
                # Create the folder if it has not been previously created
                if not os.path.exists(pat2SectionFold):
                    os.makedirs(pat2SectionFold)
                # For each section, record the file from which the training data is obtained and the columns indeces used for training the data
                f = open(pat2SectionFold + "\\SectionInformation.txt", "a")
                f.write("Currently in section {}, Csv file used for training is {}, The indeces used for training are {}".format(LED2Section,TrainingParam["Path2ii690TrainingData"],[i+2 for i in range ((LED2Section*pxlShift),(LED2Section*pxlShift)+TrainingParam["InputPerSection"])]))
                f.close()
                # Construct the path to save the weight file 
                pat2Weight = pat2SectionFold + "\\ii690{}Section{}.h5".format(LED2Section+1,section+1)
                # Based on the selected metrics, instantaite a new metrics class for every section
                if TrainingParam["Loss"] == "MAP Error":
                    loss = keras.losses.MeanAbsolutePercentageError()
                    metrics = keras.metrics.MeanAbsolutePercentageError()
                    callbackMonitorQuality = "val_mean_absolute_percentage_error"
                elif TrainingParam["Loss"] == "MS Error":
                    loss = keras.losses.MeanSquaredError()
                    metrics = keras.metrics.MeanSquaredError()
                    callbackMonitorQuality = "val_mean_squared_error"
                # Checkpoint to save the best weights
                checkpoint_callback = keras.callbacks.ModelCheckpoint(
                    filepath=pat2Weight,
                    save_weights_only=True,
                    verbose = 1,
                    monitor=callbackMonitorQuality,
                    mode='min',
                    save_best_only=True)
                # Create the model
                tempMod = SingleLayerNeuralNet(InputsPerSection=TrainingParam["InputPerSection"],HiddenUnits=TrainingParam["FirstLayerHiddenUnits"],Activation=TrainingParam["Activation"],Optimizer=keras.optimizers.Adam(lr=0.01),Loss=loss,Metrics=metrics,NoOutput=TrainingParam["NoOfOutput"])
            else:
                # Path to folder for different layers of hidden units
                pat2LayerOfHiddenUnits = pat2NumberOfOverlap + "\\TwoHidden"
                # Create the folder if it has not been previously created
                if not os.path.exists(pat2LayerOfHiddenUnits):
                    os.makedirs(pat2LayerOfHiddenUnits)
                # Path to folder containing the number of hidden units
                pat2NumberOfHiddenUnits = pat2LayerOfHiddenUnits + "\\{} First {} Second".format(TrainingParam["FirstLayerHiddenUnits"],TrainingParam["SecondLayerHiddenUnits"])
                # Create the folder if it has not been previously created
                if not os.path.exists(pat2NumberOfHiddenUnits):
                    os.makedirs(pat2NumberOfHiddenUnits)
                # At the weight destination file path, create a folder for the current section to store its weight file 
                pat2SectionFold = pat2NumberOfHiddenUnits + "\\Section {}".format(section+1)
                # Create the folder if it has not been previously created
                if not os.path.exists(pat2SectionFold):
                    os.makedirs(pat2SectionFold)
                # For each section, record the file from which the training data is obtained and the columns indeces used for training the data
                f = open(pat2SectionFold + "\\SectionInformation.txt", "a")
                f.write("Currently in section {}, Csv file used for training is {}, The indeces used for training are {}".format(LED2Section,TrainingParam["Path2ii690TrainingData"],[i+2 for i in range ((LED2Section*pxlShift),(LED2Section*pxlShift)+TrainingParam["InputPerSection"])]))
                f.close()
                # Construct the path to save the weight file 
                pat2Weight = pat2SectionFold + "\\ii690{}Section{}.h5".format(LED2Section+1,section+1)
                # Based on the selected metrics, instantaite a new metrics class for every section
                if TrainingParam["Loss"] == "MAP Error":
                    loss = keras.losses.MeanAbsolutePercentageError()
                    metrics = keras.metrics.MeanAbsolutePercentageError()
                    callbackMonitorQuality = "val_mean_absolute_percentage_error"
                elif TrainingParam["Loss"] == "MS Error":
                    loss = keras.losses.MeanSquaredError()
                    metrics = keras.metrics.MeanSquaredError()
                    callbackMonitorQuality = "val_mean_squared_error"
                # Checkpoint to save the best weights
                checkpoint_callback = keras.callbacks.ModelCheckpoint(
                    filepath=pat2Weight,
                    save_weights_only=True,
                    verbose = 1,
                    monitor=callbackMonitorQuality,
                    mode='min',
                    save_best_only=True)
                # Create the model
                tempMod = DoubleLayerNeuralNet(InputsPerSection=TrainingParam["InputPerSection"],FirstLayerHiddenUnits=TrainingParam["FirstLayerHiddenUnits"],SecondLayerHiddenUnits=TrainingParam["SecondLayerHiddenUnits"],Activation=TrainingParam["Activation"],Optimizer=keras.optimizers.Adam(lr=0.01),Loss=loss,Metrics=metrics,NoOutput=TrainingParam["NoOfOutput"])
            # Train the model
            history = tempMod.model.fit(X_train,Y_train,batch_size=int(X_train.shape[0]//2),epochs=TrainingParam["Epochs"],validation_data=(X_valid,Y_valid),callbacks=[checkpoint_callback])
            # Save the minimum validation error for the current section
            sectErr[0,section] = min(history.history[callbackMonitorQuality])
            # Clear the training session for the current section
            K.clear_session()# Checked

        #After all sections have been trained, save the error values
        np.savetxt(pat2NumberOfHiddenUnits+"\\MinError.csv", sectErr, delimiter=",")