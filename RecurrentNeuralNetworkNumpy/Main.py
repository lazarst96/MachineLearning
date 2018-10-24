from NeuralNetwork import *



model = NeuralNetwork(input_size=3,optimizer=Adam(0.001),lossfunction=CrossEntropyWithSigmoid())
model.addLayer(layer_size=20, bias=False, activationfunction=Identity())
model.addLayer(layer_size=10, bias=False, activationfunction=Identity())
model.addLayer(layer_size=10, bias=False, activationfunction=Identity())
model.addLayer(layer_size=10, bias=False, activationfunction=Identity())
model.addLayer(layer_size=1, bias=True, activationfunction=Sigmoid())

#Nisam primenjivao na neki konkretan problem, ovo je samo primer za kako se kreira arhitektura.

