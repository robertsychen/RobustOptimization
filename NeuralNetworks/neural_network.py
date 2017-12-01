import numpy as np
import tensorflow as tf #Note: uses version 0.11
import copy
import time
import pickle

from noise_application import DirectNoise

class NeuralNetwork(object):
    '''
    Building the standard feed-forward, fully-connected network architecture as well as the special network architecture
    for the Composite Oracle Method. Implements functionality for mini-batch gradient descent training as well as
    evaluation/prediction on the test data.
    '''
    def __init__(self,
                 train_dataset, 
                 train_labels, 
                 valid_dataset, 
                 valid_labels,
                 image_size=28, 
                 num_labels=10, 
                 batch_size=100, 
                 num_hidden_layers=1,
                 num_hidden_nodes=1024, 
                 optimizer_type='GradientDescent',
                 optimizer_params={'learning_rate':0.5},
                 is_custom_weights=False,
                 custom_weights=None,
                 is_composite_method=False,
                 num_noise_types=None,
                 noise_dist=None):
        self.train_dataset = train_dataset
        self.train_labels = train_labels
        self.valid_dataset = valid_dataset
        self.valid_labels = valid_labels

        self.image_size = image_size #number of pixels along side of each image; assumes square images
        self.num_labels = num_labels #total number of classes
        self.batch_size = batch_size #number of training images in each mini-batch
        self.num_hidden_layers = num_hidden_layers #currently must be 1. could easily implement additional layers
        self.num_hidden_nodes = num_hidden_nodes #number of nodes per hidden layer
        self.optimizer_type = optimizer_type #optimizer being used; must be 'GradientDescent' but could easily implement others
        self.optimizer_params = optimizer_params #dictionary of optimizer params. For 'GradientDescent', needs a 'learning_rate'
        
        self.is_custom_weights = is_custom_weights #whether to initialize weights with specified values instead of using random initialization
        self.custom_weights = custom_weights #format: list of length (num_hidden_layers + 1)*2 of numpy arrays (see use below)

        #Composite Oracle Method specific:
        self.is_composite_method = is_composite_method
        self.num_noise_types = num_noise_types 
        self.noise_dist = noise_dist
    
        #Set up graph structure.
        self.graph = tf.Graph()
        with self.graph.as_default():
            if not self.is_composite_method: #standard neural network setup
                self.tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
                self.tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
                self.tf_valid_dataset = tf.constant(self.valid_dataset)
                
                if num_hidden_layers == 1:
                    if not is_custom_weights:
                        self.weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_hidden_nodes]))
                        self.biases1 = tf.Variable(tf.zeros([num_hidden_nodes]))
                        self.weights2 = tf.Variable(tf.truncated_normal([num_hidden_nodes, num_labels]))
                        self.biases2 = tf.Variable(tf.zeros([num_labels]))
                    else:
                        self.weights1 = tf.Variable(custom_weights[0])
                        self.biases1 = tf.Variable(custom_weights[1])
                        self.weights2 = tf.Variable(custom_weights[2])
                        self.biases2 = tf.Variable(custom_weights[3])
                    self.lay1_train = tf.nn.relu(tf.matmul(self.tf_train_dataset, self.weights1) + self.biases1)
                    self.logits = tf.matmul(self.lay1_train, self.weights2) + self.biases2
                    self.lay1_valid = tf.nn.relu(tf.matmul(tf.cast(self.tf_valid_dataset, tf.float32), self.weights1) + self.biases1)
                    self.valid_logits = tf.matmul(self.lay1_valid, self.weights2) + self.biases2
                    self.valid_prediction = tf.nn.softmax(self.valid_logits)
                    self.valid_loss = tf.mul(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.valid_logits, self.valid_labels)), -1.0) 

                    self.train_prediction = tf.nn.softmax(self.logits)
                    self.indiv_loss = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.tf_train_labels)
                    self.loss = tf.mul(tf.reduce_mean(self.indiv_loss), -1.0)
                    
                else:
                    raise ValueError('This number of hidden layers not currently supported.')

            else: #self.is_composite_method == True; special network architecture setup for Composite Method Oracle
                self.tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size, num_noise_types))
                self.tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
                self.tf_valid_dataset = tf.constant(self.valid_dataset)

                if num_hidden_layers == 1:
                    if not is_custom_weights:
                        self.weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_hidden_nodes]))
                        self.biases1 = tf.Variable(tf.zeros([num_hidden_nodes]))
                        self.weights2 = tf.Variable(tf.truncated_normal([num_hidden_nodes, num_labels]))
                        self.biases2 = tf.Variable(tf.zeros([num_labels]))
                    else:
                        self.weights1 = tf.Variable(custom_weights[0])
                        self.biases1 = tf.Variable(custom_weights[1])
                        self.weights2 = tf.Variable(custom_weights[2])
                        self.biases2 = tf.Variable(custom_weights[3])

                    self.lay1_train = []
                    self.logits = []
                    self.train_prediction =[]
                    self.indiv_loss = []
                    for i in xrange(num_noise_types):
                        self.lay1_train.append(tf.nn.relu(tf.matmul(self.tf_train_dataset[:,:,i], self.weights1) + self.biases1))
                        self.logits.append(tf.matmul(self.lay1_train[i], self.weights2) + self.biases2)
                        self.train_prediction.append(tf.nn.softmax(self.logits[i]))
                        self.indiv_loss.append(tf.scalar_mul(tf.cast(tf.constant(self.noise_dist[i]), tf.float32), 
                                tf.nn.softmax_cross_entropy_with_logits(self.logits[i], self.tf_train_labels)))

                    #Combine loss over all data and noise types.
                    self.combined_indiv_loss = sum(self.indiv_loss)
                    self.loss = tf.mul(tf.reduce_mean(self.combined_indiv_loss), -1.0) #make loss negated to fit robopt maximization framework

                    #For validation (aka testing)
                    self.lay1_valid = tf.nn.relu(tf.matmul(self.tf_valid_dataset, self.weights1) + self.biases1)
                    self.valid_logits = tf.matmul(self.lay1_valid, self.weights2) + self.biases2
                    self.valid_prediction = tf.nn.softmax(self.valid_logits)
                    
                    self.valid_loss = tf.mul(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.valid_logits, self.valid_labels)), -1.0) 
                    #^make loss negated to fit robopt maximization framework
                    
                else:
                    raise ValueError('This number of hidden layers not currently supported.')
            
            if optimizer_type == 'GradientDescent':
                self.optimizer = tf.train.GradientDescentOptimizer(optimizer_params['learning_rate']).minimize(tf.mul(self.loss, -1.0))
            else:
                raise ValueError('Optimizer type is not currently supported.')
                
    def __accuracy(self, predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
            
    def train(self, num_steps, variable_storage_file_name, verbose=True, noise_type='none'):    
        #Allows for noisy training, where certain type of noise is added to train set beforehand.
        #Note noise_type is single noise if not Composite Method; if Composite Method, it's list of all the noises.
        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run(session=session)

            if not self.is_composite_method:
                this_train_dataset = DirectNoise(copy.deepcopy(self.train_dataset), noise_type).apply_noise()
                
                #Do actual training.
                for step in range(num_steps):
                    index_subset = np.random.choice(self.train_labels.shape[0], size=self.batch_size)
                    batch_data = this_train_dataset[index_subset, :]
                    batch_labels = self.train_labels[index_subset, :]
                    feed_dict = {self.tf_train_dataset : batch_data, self.tf_train_labels : batch_labels}
                    _, l, predictions = session.run([self.optimizer, self.loss, self.train_prediction], feed_dict=feed_dict)
                    if verbose and (step % (num_steps/10) == 0):
                        print("Minibatch loss at step %d: %f" % (step, l))

            else: #self.is_composite_method == True
                this_train_dataset = copy.deepcopy(self.train_dataset)
                all_noise_datasets = np.zeros((this_train_dataset.shape[0], this_train_dataset.shape[1], self.num_noise_types))
                counter = 0
                for name in noise_type:
                    all_noise_datasets[:,:,counter] = DirectNoise(this_train_dataset, name).apply_noise()
                    counter += 1

                for step in xrange(num_steps):
                    #Select desired index subset for mini-batch (same across all noise types).
                    index_subset = np.random.choice(self.train_labels.shape[0], size=self.batch_size)
                    all_batch_data = all_noise_datasets[index_subset,:,:]
                    batch_labels = self.train_labels[index_subset, :]

                    #Run training iteration.
                    feed_dict = {self.tf_train_dataset : all_batch_data, self.tf_train_labels : batch_labels}
                    _, l, predictions = session.run([self.optimizer, self.loss, self.train_prediction], feed_dict=feed_dict)

            saver = tf.train.Saver() #This saving behavior allows us to train multiple versions of the model using the same class.
            save_path = saver.save(session, variable_storage_file_name)
            
            #Format the trained weights so we can return them.
            trained_weights = [self.weights1.eval(), self.biases1.eval()]
            if self.num_hidden_layers >= 1:
                trained_weights.append(self.weights2.eval())
                trained_weights.append(self.biases2.eval())
            if self.num_hidden_layers >= 2:
                trained_weights.append(self.weights3.eval())
                trained_weights.append(self.biases3.eval())
            
            evaluated_val_loss = self.valid_loss.eval()
            return evaluated_val_loss, trained_weights
        
    def test(self, variable_storage_file_name, new_dataset=0, new_labels=0):
        #If new_dataset is 0 and new_labels is 0: uses the validation set and labels to predict and score.
        #If new_dataset isn't 0 and new_labels is 0: uses the new test set to predict.
        #If new_dataset isn't 0 and new_labels isn't 0: uses the new test set and labels to predict and score.
        #Note: new_dataset & new_labels must have same # of data points as self.valid_dataset, self.valid_labels for this to run.
        resulting_lossval = None
        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run(session=session)
            saver = tf.train.Saver()
            saver.restore(session, variable_storage_file_name)     
            
            this_dataset = None
            this_labels = None
            if type(new_dataset) is not int:
                this_dataset = new_dataset
                this_labels = new_labels
            else:
                this_dataset = self.valid_dataset
                this_labels = self.valid_labels   
            
            feed_dict_clean = {self.tf_valid_dataset: this_dataset}
            valid_lossval = session.run(self.valid_loss, feed_dict=feed_dict_clean)
            valid_y = session.run(self.valid_prediction, feed_dict=feed_dict_clean)
            resulting_lossval = None if type(new_dataset) is not int and type(new_labels) is int else valid_lossval
        return resulting_lossval, valid_y
    
    def test_noisy_version(self, variable_storage_file_name, noise_type='none'):
        noisy_dataset = DirectNoise(copy.deepcopy(self.valid_dataset), noise_type).apply_noise()
        return self.test(variable_storage_file_name, noisy_dataset, self.valid_labels)