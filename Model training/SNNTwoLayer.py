import tensorflow  as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers

class DoubleLayerNeuralNet:
    def __init__(self,InputsPerSection,FirstLayerHiddenUnits,SecondLayerHiddenUnits,Activation,Optimizer,Loss,Metrics,WeightInitializer='random_normal',NoOutput=2):
        # Create the model
        self.model = Sequential()
        self.model.add(Dense(FirstLayerHiddenUnits,batch_input_shape=(None,InputsPerSection),activation=Activation,kernel_initializer=WeightInitializer))
        self.model.add(Dense(SecondLayerHiddenUnits,activation=Activation,kernel_initializer=WeightInitializer))
        self.model.add(Dense(NoOutput))
        self.model.compile(loss=Loss,optimizer=Optimizer,metrics=[Metrics])
        self.model.summary()

    def predict(self,input_tensor,pat2weight):
        self.model.load_weights(pat2weight)
        return self.model.predict(input_tensor)

