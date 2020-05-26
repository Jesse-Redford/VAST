import keras
from keras.models import load_model
from matplotlib import pyplot
from math import cos, sin, atan
from palettable.tableau import Tableau_10
from time import localtime, strftime
import numpy as np
import os
import matplotlib
from matplotlib import rc
#matplotlib.rcParams['font.sans-serif'] = "Helvetica"
#matplotlib.rcParams['font.style'] = "italic"
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Cambria Math'] + matplotlib.rcParams['font.serif']

class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, neuron_radius, id=-1):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)
        
############### ADD NUMBER TO EACH NEURON #######################################
        
       # pyplot.gca().text(self.x, self.y-0.15, str(id), size=8, ha='center')
       # pyplot.gca().text(self.x, self.y-0.15, str(id), size=8, ha='center')
       
##################################################################################

class Layer():
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer):
        self.vertical_distance_between_layers =  25 #6  # make sure this matches value in draw function NNclass
        self.horizontal_distance_between_neurons = 2
        self.neuron_radius = 0.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.horizontal_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, weight=0.4, textoverlaphandler=None):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)

        # assign colors to lines depending on the sign of the weight
        color=Tableau_10.mpl_colors[0]
        if weight > 0: color=Tableau_10.mpl_colors[1]

        # assign different linewidths to lines depending on the size of the weight
        abs_weight = abs(weight)        
        #if abs_weight > 0.5: 
         #   linewidth = 10*abs_weight
        #elif abs_weight > 0.8: 
         #   linewidth =  100*abs_weight
        #else:
            #linewidth = abs_weight
        linewidth = abs_weight*1
        # draw the weights and adjust the labels of weights to avoid overlapping
        if abs_weight > 0.0 or abs_weight < 0 : 
        #if abs_weight > 0.5: 
            # while loop to determine the optimal locaton for text lables to avoid overlapping
            index_step = 2
            num_segments = 10   
            txt_x_pos = neuron1.x - x_adjustment+index_step*(neuron2.x-neuron1.x+2*x_adjustment)/num_segments
            txt_y_pos = neuron1.y - y_adjustment+index_step*(neuron2.y-neuron1.y+2*y_adjustment)/num_segments
            while ((not textoverlaphandler.getspace([txt_x_pos-0.5, txt_y_pos-0.5, txt_x_pos+0.5, txt_y_pos+0.5])) and index_step < num_segments):
                index_step = index_step + 1
                txt_x_pos = neuron1.x - x_adjustment+index_step*(neuron2.x-neuron1.x+2*x_adjustment)/num_segments
                txt_y_pos = neuron1.y - y_adjustment+index_step*(neuron2.y-neuron1.y+2*y_adjustment)/num_segments

            # print("Label positions: ", "{:.2f}".format(txt_x_pos), "{:.2f}".format(txt_y_pos), "{:3.2f}".format(weight))
           
############ ADD WEIGHT VALUES TO PLOT, uincomment two lines below ###################    
       
           # a=pyplot.gca().text(txt_x_pos, txt_y_pos, "{:3.2f}".format(weight), size=8, ha='center')
            #a.set_bbox(dict(facecolor='white', alpha=0))
           
#################################################################   

  
            # print(a.get_bbox_patch().get_height())

        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), (neuron1.y - y_adjustment, neuron2.y + y_adjustment), linewidth=linewidth, color=color)
        pyplot.gca().add_line(line)

    def draw(self, layer_info = None,layerType=0, weights=None, textoverlaphandler=None):
        j=0 # index for neurons in this layer
        for neuron in self.neurons:            
            i=0 # index for neurons in previous layer
            neuron.draw( self.neuron_radius, id=j+1 )
            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, weights[i,j], textoverlaphandler)
                    i=i+1
            j=j+1
            
     ######## Write the number nodes and activation function beside each layer of the network #########       
       
        #x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons  
        x_text = self.number_of_neurons_in_widest_layer  * self.horizontal_distance_between_neurons 
        pyplot.text(x_text, self.y, str(layer_info), fontsize = 15)
            
        
############ write Text beside each Layer #######################################
       # x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons
       # if layerType == 0:
        #    pyplot.text(x_text, self.y, 'Input Layer', fontsize = 12)
        #elif layerType == -1:
         #   pyplot.text(x_text, self.y, 'Output Layer', fontsize = 12)
       # else:
        #    pyplot.text(x_text, self.y, 'Hidden Layer '+str(layerType), fontsize = 12)
###########################################################################################

# A class to handle Text Overlapping
# The idea is to first create a grid space, if a grid is already occupied, then
# the grid is not available for text labels.
class TextOverlappingHandler():
    # initialize the class with the width and height of the plot area
    def __init__(self, width, height, grid_size=0.2):
        self.grid_size = grid_size
        self.cells = np.ones((int(np.ceil(width / grid_size)), int(np.ceil(height / grid_size))), dtype=bool)

    # input test_coordinates(bottom left and top right), 
    # getspace will tell you whether a text label can be put in the test coordinates
    def getspace(self, test_coordinates):
        x_left_pos = int(np.floor(test_coordinates[0]/self.grid_size))
        y_botttom_pos = int(np.floor(test_coordinates[1]/self.grid_size))
        x_right_pos = int(np.floor(test_coordinates[2]/self.grid_size))
        y_top_pos = int(np.floor(test_coordinates[3]/self.grid_size))
        if self.cells[x_left_pos, y_botttom_pos] and self.cells[x_left_pos, y_top_pos] \
        and self.cells[x_right_pos, y_top_pos] and self.cells[x_right_pos, y_botttom_pos]:
            for i in range(x_left_pos, x_right_pos):
                for j in range(y_botttom_pos, y_top_pos):
                    self.cells[i, j] = False

            return True
        else:
            return False

class NeuralNetwork():
    def __init__(self, number_of_neurons_in_widest_layer,model_name,optimizer,optimizer_parameters,layer_info ): #model_name = 'model', model_config = '', path=''):
        #self.path = path
        #self.model_config = model_config
        #self.model_name = model_name
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []
        self.layertype = 0
        self.layer_info = layer_info
        self.model_name = model_name
        self.optimizer = optimizer
        self.optimizer_parameters = optimizer_parameters
      

    def add_layer(self, number_of_neurons ):
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer)
        self.layers.append(layer)

    def draw(self, weights_list=None): #, model_name = 'model', model_config ='', path=''):
        # vertical_distance_between_layers and horizontal_distance_between_neurons are the same with the variables of the same name in layer class
        vertical_distance_between_layers = 25 #6
        horizontal_distance_between_neurons = 2
        overlaphandler = TextOverlappingHandler(\
            self.number_of_neurons_in_widest_layer*horizontal_distance_between_neurons,\
            len(self.layers)*vertical_distance_between_layers, grid_size=0.2 )
        """ Format Layer info """
       
        #layer_info = str('$L_{' + str(self.layer_info[i][1])+'} ^{' + str(self.layer_info[i][0]) + '}$')
        #layer_info = str(r'$\Re \in $' + str(self.layer_info[i][0])) + '  ' + '$A_{' + str(self.layer_info[i][1])  +'}$'
         
        
        pyplot.figure(figsize=(12, 9))

        for i in range( len(self.layers) ):
            layer = self.layers[i] 
            layer_info = str('$\mathcal{L}_{' + str(self.layer_info[i][1])+'} ^{' + str(self.layer_info[i][0]) + '}$')
            #layer_info = str(self.layer_info[i][0]) + '  ' + str(self.layer_info[i][1])
                                     
            if i == 0:
                layer.draw( layer_info, layerType=0 ) 
            elif i == len(self.layers)-1:
                layer.draw( layer_info,layerType=-1, weights=weights_list[i-1], textoverlaphandler=overlaphandler)
            else:
                layer.draw(layer_info,layerType=i, weights=weights_list[i-1], textoverlaphandler=overlaphandler)
    
        pyplot.axis('scaled')
        pyplot.axis('off')
        
        
        
       # pyplot.title( 'Neural Network architecture', fontsize=15 )
        #title =  str(self.model_name) + '\n' + str(self.optimizer)+ '  ' + str(self.optimizer_parameters)
       
        pyplot.title(str(self.model_name), fontsize=12,fontname='Cambria Math' ) #,  bbox={'facecolor':'silver', 'alpha':0.5, 'pad':4})
        #pyplot.suptitle(title, fontsize=15 ,  bbox={'facecolor':'silver', 'alpha':0.5, 'pad':4})
        #pyplot.title(self.model_config  + '\n' + '' + '\n' + self.model_name, fontsize= 5 , wrap = True)
       
        #figureName=self.model_name+strftime("%Y%m%d_%H%M%S", localtime())+'.png'
        #pyplot.tight_layout
        #pyplot.savefig(self.path+figureName, dpi=300, bbox_inches="tight")
        #pyplot.text(10,10,self.model_config)
        #pyplot.savefig(str(self.model_name)+strftime("%Y%m%d_%H%M%S", localtime())+'.png',dpi=1200)
        pyplot.savefig(str(self.model_name)+strftime("%Y%m%d_%H%M%S", localtime())+'.png',format='png',bbox_inches='tight')
        #pyplot.savefig(self.path+figureName, dpi=300, bbox_inches=0)
      
        pyplot.show()
        
import keras.backend as K
import tensorflow as tf

class DrawNN():
    # para: neural_network is an array of the number of neurons 
    # from input layer to output layer, e.g., a neural network of 5 nerons in the input layer, 
    # 10 neurons in the hidden layer 1 and 1 neuron in the output layer is [5, 10, 1]
    # para: weights_list (optional) is the output weights list of a neural network which can be obtained via classifier.coefs_
   # def __init__( self, neural_network, weights_list=None ):
    def __init__( self, model): #, weights_list=None, model_name = 'Neural Network Architecture', model_config = '', path = ''):
        
        """ Load Model.h5 and extract details, ei: name, shape, weights, layer info, optimizer, hyperparameters, .... """
        self.model_name = str(model)
        model = load_model(str(model))
       
        
        
        model_shape = []
        weights = []
        layer_info = []
        config = model.get_config() 
        model_shape.append(model.inputs[0][0].shape[0])
        
        i=0
        for layer in model.layers:
            model_shape.append(layer.output_shape[1])
            weights.append(np.array(np.asmatrix(model.layers[i].get_weights()[0])))
            i += 1
               
        layers = config['layers']
        layer_info.append([layers[0]['config']['batch_input_shape'][1], ''])
        for layer in layers:
            if layer['class_name'] == 'Dense':
                layer_info.append([layer['config']['units'], layer['config']['activation']])
 
        
        self.optimizer = str(model.optimizer).split('.')[2].split()[0]
        self.optimizer_parameters = model.optimizer.get_config()  # optomizer settings = lr, rho, decay, eps
       # self.learning_rate = K.eval(model.optimizer.lr)
        self.neural_network = model_shape
        self.weights_list = weights
        self.layer_info = layer_info
        
        
        # if weights_list is none, then create a uniform list to fill the weights_list
        if self.weights_list is None:
            weights_list=[]
            for first, second in zip(neural_network, neural_network[1:]):
                tempArr = np.ones((first, second))*0.4
                weights_list.append(tempArr)
            self.weights_list = weights_list
        
    def draw( self ):
        model_name = self.model_name
        optimizer = self.optimizer
        optimizer_parameters =  self.optimizer_parameters
        layer_info = self.layer_info
        widest_layer = max( self.neural_network )
        network = NeuralNetwork( widest_layer,model_name,optimizer,optimizer_parameters,layer_info) 
        for l in self.neural_network:
            network.add_layer(l)
        network.draw(self.weights_list) 