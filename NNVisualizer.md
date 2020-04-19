# Looking Under the hood of a Keras model, Matrix Multiplication and Visualization for Deep-Qnetwork

Creating and training a Keras model is relativley when you follow along with tutorials online. However, most of the examples you find 
online treat a NN as somewhat of a blackbox. Using just a few lines of code, we look at the matrix multiplication assosicated with
Forward Propagation and create a visualization tool which allows us to look under the hood of Neural Network during training.

# Creating a Deep-Qnetwork in Keras
For this example we will assume that the state of our agent will consist of a 3 dimensional vector S = [s1,s2,s3], 
where -inf < s1,s2,s3 < inf. The avalible action space for the agent will consist of a 1-D action list A, with each entry containing 
a differnt combination of control outputs (a1,a2,a3) we can send to the system. Based on this information, the architecture for our Deep-Qnetwork
will consist of a 3 node input layer, followed by two hidden layers, and an n-node output layer. Where n is the number of entries in our 1-D action list, 
such that the NN can output a Q estimate for each action, using only one forward pass. We will use 3 forward propagation methods 
to demonstate how to preform forward propogation with matrix multiplication


<details>
  <summary>Click here to see example code</summary>
      
      import keras
      from keras.models import Sequential
      from keras.layers import Dense
      from keras.optimizers import RMSprop
      import itertools
      import numpy as np
      
      # Define Example of State input the NN can expect, and a list of actions 
      S = (0,0,0)
      A = list(itertools.product(range(0,2,1),range(0,2,1)))
      
      # Build NN Model
      layer1_neurons = 2
      layer2_neurons = 2
      model = Sequential()
      model.add(Dense(layer1_neurons, input_dim = len(S), activation='relu',kernel_initializer='normal'))
      model.add(Dense(layer2_neurons, activation='relu',kernel_initializer='normal'))
      model.add(Dense(output_dim = len(actions), activation='relu',kernel_initializer='normal'))
      model.compile(loss='mse', optimizer=RMSprop(lr=.1))
      
      # You might be used to using model.predict(state), to get the models Q-estimates for each action given a state input. 
      # However, in the example below we will look at what is actually happending when we call model.predict(state)
      # Lets start by extracting the weights and bias terms for each layer in our model and representing them as matrcies.
    
      w1 = first_layer_weights = np.asmatrix(model.layers[0].get_weights()[0])
      b1 = first_layer_biases  = np.matrix(model.layers[0].get_weights()[1])

      w2 = second_layer_weights = np.matrix(model.layers[1].get_weights()[0])
      b2 = second_layer_biases  = np.matrix(model.layers[1].get_weights()[1])

      w3 = third_layer_weights = np.matrix(model.layers[2].get_weights()[0])
      b3 = third_layer_biases  = np.matrix(model.layers[2].get_weights()[1])
      
      print(w1.shape, b1.shape)
      print(w2.shape, b2.shape)
      print(w3.shape, b3.shape)
      
      # Now lets create a method for computing the relu activation, which we defined for each of the keras models layers.
      def relu(Z):
          return np.maximum(0, Z)
      
      # Now lets compute the models Q_estimates, based on an input state = (1,1,1) 
      state = (1,1,1)
      
      # To pass this state to our NN we need to format it as shown
      input_state = (np.asarray(state).reshape(1, len(state)))
      
      # Method 1
      Q_estimates_method1 = model.predict(state)
      
      # Method 2
      Z = np.dot(state,w1)+b1
      H = relu(Z)
      Z1 = np.dot(H,w2)+b2
      H1 = relu(Z1)
      Z2 = np.dot(H1,w3)+b3
      Q_estimates_method2 = relu(Z2)
      
      # Method 3
      Q_estimates_method3 = relu(np.dot(relu(np.dot(relu(np.dot(state,w1)+b1),w2)+b2),w3)+b3)
          
      
      # Results for each method
      print(' Q_estimates_method1', Q_estimates_method1)
      print(' Q_estimates_method2', Q_estimates_method2)
      print(' Q_estimates_method3', Q_estimates_method3)
      
      

  

</details>

As you can see, each method generates the same output. 
We use matrix multiplecation to compute a single forward pass of our network.
if the activation for each layer of our keras model was linear, we could simple remove the relu activation 
from method 2 and 3, otherwise we would just have to make sure the activation function matches what we defined 
when we were building the model.
      
      
# Visualizing the Learning Process

Now that you understand the matrix multiplication assosiated with forward propogration, lets look at how 
we can visualize what is going on under the hood of a Deep Qnetwork as we train it to give a particular Q_estimate for 
a particular input. Before proceeding with the example, make sure you download the VisualizeNN.py file. 


<details>
  <summary>Click here to see example code</summary>
    
    import VisualizeNN as VisNN
    import os
    import imageio
    import sys
    
    model_name = ' Deep-Qnetwork'
    some_state = (1,1,1)
    desired_output = (10,-10,5,1)
    num_of_back_propagations = 10
    image_path = r'C:/Users/Jesse/Desktop/Snakebot_Backup-3-30-2020/visualize-neural-network-master/NN_Images/' 
    animation_path = r'C:/Users/Jesse/Desktop/Snakebot_Backup-3-30-2020/visualize-neural-network-master/NN_Animations/'
    
    # Recall the size of each of our models layers
    input_layer = 3
    hidden_layer1 = 2
    hidden_layer1 = 2
    output_layer = 4
    
    input_state = (np.asarray(state).reshape(1, len(state)))
    desired_Qestimates = (np.asarray(desired_output).reshape(1, len(desired_output)))
    
    
    for i in range(num_of_back_propagations):
        w1 = np.asmatrix(model.layers[0].get_weights()[0])
        w2 = np.asmatrix(model.layers[1].get_weights()[0])
        w3 = np.asmatrix(model.layers[2].get_weights()[0])
        weights = [np.array(w1),np.array(w2),np.array(w3)]
        
        # Create Diagram.png file of NN with current weights
        network=VisNN.DrawNN([input_layer,hidden_layer1,hidden_layer2,output_layer], weights,model_name,image_path) 
        network.draw()

        model.train_on_batch(input_state,desired_Qestimates)
    
    # Create Animation from png files
    images = []
    for file_name in os.listdir(image_path):
      if file_name.endswith('.png'):
        file_path = os.path.join(image_path, file_name)
        images.append(imageio.imread(file_path))
    imageio.mimsave(animation_path+'movie.gif', images)
    

</details>


# Putting it all together, make sure you edit the 
<details>
  <summary>Run Code After Downloading VisualizeNN.py file </summary>
      
      import keras
      from keras.models import Sequential
      from keras.layers import Dense
      from keras.optimizers import RMSprop
      import itertools
      import numpy as np
      
      # Define Example of State input the NN can expect, and a list of actions 
      S = (0,0,0)
      A = list(itertools.product(range(0,2,1),range(0,2,1)))
      
      # Build NN Model
      layer1_neurons = 2
      layer2_neurons = 2
      model = Sequential()
      model.add(Dense(layer1_neurons, input_dim = len(S), activation='relu',kernel_initializer='normal'))
      model.add(Dense(layer2_neurons, activation='relu',kernel_initializer='normal'))
      model.add(Dense(output_dim = len(actions), activation='relu',kernel_initializer='normal'))
      model.compile(loss='mse', optimizer=RMSprop(lr=.1))
      
    import VisualizeNN as VisNN
    import os
    import imageio
    import sys
    
    model_name = ' Deep-Qnetwork'
    some_state = (1,1,1)
    desired_output = (10,-10,5,1)
    num_of_back_propagations = 10
    image_path = r'C:/Users/Jesse/Desktop/Snakebot_Backup-3-30-2020/visualize-neural-network-master/NN_Images/' 
    animation_path = r'C:/Users/Jesse/Desktop/Snakebot_Backup-3-30-2020/visualize-neural-network-master/NN_Animations/'
    
    # Recall the size of each of our models layers
    input_layer = 3
    hidden_layer1 = 2
    hidden_layer1 = 2
    output_layer = 4
    
    input_state = (np.asarray(state).reshape(1, len(state)))
    desired_Qestimates = (np.asarray(desired_output).reshape(1, len(desired_output)))
    
    
    for i in range(num_of_back_propagations):
        w1 = np.asmatrix(model.layers[0].get_weights()[0])
        w2 = np.asmatrix(model.layers[1].get_weights()[0])
        w3 = np.asmatrix(model.layers[2].get_weights()[0])
        weights = [np.array(w1),np.array(w2),np.array(w3)]
        
        # Create Diagram.png file of NN with current weights
        network=VisNN.DrawNN([input_layer,hidden_layer1,hidden_layer2,output_layer], weights,model_name,image_path) 
        network.draw()

        model.train_on_batch(input_state,desired_Qestimates)
    
    # Create Animation from png files
    images = []
    for file_name in os.listdir(image_path):
      if file_name.endswith('.png'):
        file_path = os.path.join(image_path, file_name)
        images.append(imageio.imread(file_path))
    imageio.mimsave(animation_path+'movie.gif', images)
    

</details>
    
