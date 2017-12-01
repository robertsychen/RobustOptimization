import numpy as np
import tensorflow as tf
import copy
import time
import pickle
from scipy import stats

from neural_network import NeuralNetwork
from noise_application import DirectNoise

class RobustOptimizer(object):
    '''
    Contains all the core functionality for running the robust optimization algorithm (Hybrid Oracle & Composite Oracle) and baselines.
    Performs test loss evaluation (Individual Bottleneck Loss & softmax-based Ensemble Bottleneck Loss) as well as test set prediction.
    Note: use more user-friendly wrapper in experiment_runners.py to run robust optimization experiments.
    '''
    def __init__(self,
                 dataset1,
                 labels1,
                 dataset2,
                 labels2,
                 dataset3,
                 labels3,
                 noise_names,
                 oracle_method,
                 update_method,
                 training_num_iters=500,
                 scale_bound=3.0,
                 initial_dist_over_noise=None):
        self.dataset1 = dataset1 #for training network
        self.labels1 = labels1 #for training network
        self.dataset2 = dataset2 #for updating distribution over corruption types
        self.labels2 = labels2 #for updating distribution over corruption types
        self.dataset3 = dataset3 #for evaluating test set performance
        self.labels3 = labels3 #for evaluating test set performance
        self.num_classes = self.labels1.shape[1]

        self.noise_names = noise_names
        self.num_noise_types = len(noise_names)

        self.oracle_method = oracle_method #'hybrid' or 'composite'
        self.update_method = update_method #is 'uniform_fixed' or a float. If float, then float is the nu used in MWU. (ex. 0.5 in the paper)
        self.training_num_iters = training_num_iters #how long to train each neural network

        self.dist_over_noise_scoring_sum = [0.0] * self.num_noise_types
        self.dist_over_noise = None
        if initial_dist_over_noise is not None:
            self.dist_over_noise = initial_dist_over_noise
        else:
            self.dist_over_noise = np.asarray([1.0/self.num_noise_types] * self.num_noise_types)
        self.all_dist_over_noise = [self.dist_over_noise]

        self.chosen_weights = []
        self.train_info = []
        self.test_info = {}
        self.iter_num = None #used to populate self.train_info

        self.scale_bound = scale_bound #value for scaling functions to within [0,1] to match theorems from the paper (no impact on performance)
        self.most_extreme_loss = 0.0 #to determine whether scale_bound successfully mapped all function values to [0,1]

    ############# Helper functions not related to robust optimization. ###############

    def __normalize_dist(self, dist):
        dist = np.asarray(dist)
        return dist / dist.sum()

    def __cross_entropy(self, dist1, dist2):
        #dist1 is prediction, dist2 is true labels
        result = 0.0
        small_float = 1.0e-307
        for i in xrange(dist1.shape[0]):
            result += (dist2[i] * np.log(dist1[i] + small_float))
        return result

    def __mean_cross_entropy(self, dists1, dists2):
        assert(dists1.shape == dists2.shape)
        num = dists1.shape[0]
        total = 0.0
        for i in xrange(num):
            total += self.__cross_entropy(dists1[i,:], dists2[i,:])
        return total / num
    
    def __weighted_sum(self, data, dist):
        #Compute weighted sum for variety of different Python objects.
        normalized_dist = np.asarray(self.__normalize_dist(dist))
        result = None
        if type(data[0]) == type(1.0) or type(data[0]) == type(np.zeros(2)[0]):
            result = 0.0
            for i in xrange(normalized_dist.shape[0]):
                result += (normalized_dist[i] * data[i])
        elif type(data[0]) == type(np.eye(2)):
            result = np.zeros(data[0].shape)
            for i in xrange(normalized_dist.shape[0]):
                result += (normalized_dist[i] * data[i])
        elif type(data[0]) == type([np.eye(2)]):
            scaled_data = copy.deepcopy(data)
            for i in xrange(len(data)):
                #scaled_data[i] is list of numpy arrays.
                for j in xrange(len(scaled_data[i])):
                    scaled_data[i][j] = scaled_data[i][j] * dist[i]
            #Sum the scaled components.
            result = scaled_data[0]
            for i in xrange(1,len(data)):
                for j in xrange(len(scaled_data[i])):
                    result[j] = result[j] + scaled_data[i][j]
        else:
            raise ValueError("Unsupported type")
        return result

    ############# Helper functions related to robust optimization. ###############

    def __distributional_oracle(self, dist):
        if self.oracle_method == 'hybrid':
            return self.__oracle_hybrid(dist)
        elif self.oracle_method == 'composite':
            return self.__oracle_composite(dist)
        else:
            raise ValueError("This oracle method is not currently supported.")

    def __oracle_composite(self, dist):
        #Note: the composite behavior is implemented in NeuralNetwork
        network = self.__create_neural_network(self.dataset1, self.labels1, self.dataset1, self.labels1,
                is_custom_weights=False, custom_weights=None, is_composite_method=True, num_noise_types=self.num_noise_types, noise_dist=dist)
        _, this_weight = network.train(num_steps = self.training_num_iters, variable_storage_file_name = 'mod', verbose=False,
                     noise_type=self.noise_names)
        return this_weight

    def __oracle_hybrid(self, dist):
        this_dataset = self.__perturb_data_using_noise_mixture(self.dataset1, dist)
        network = self.__create_neural_network(this_dataset, self.labels1, this_dataset, self.labels1, 
                is_custom_weights=False, custom_weights=None)
        _, this_weight = network.train(num_steps = self.training_num_iters, variable_storage_file_name = 'mod', verbose=False,
                     noise_type='none')
        return this_weight

    def __perturb_data_using_noise_mixture(self, initial_data, dist):
        data = copy.deepcopy(initial_data)
        noise_assignments = np.random.choice(self.num_noise_types, size=data.shape[0], replace=True, p=dist)
        for i in xrange(data.shape[0]):
            this_initial_data = copy.deepcopy(initial_data[i,:])
            this_initial_data = np.reshape(this_initial_data, (1, this_initial_data.shape[0]))
            data[i,:] = DirectNoise(this_initial_data, self.noise_names[noise_assignments[i]]).apply_noise()
        return data
        
    def __custom_score(self, weights, data, labels, noise_type):
        network = self.__create_neural_network(data, labels, data, labels, is_custom_weights=True, custom_weights=weights)
        network.train(num_steps = 0, variable_storage_file_name = 'mod', verbose=False)
        lossval = network.test_noisy_version('mod', noise_type=noise_type)[0]
        return lossval

    def __custom_predict(self, weights, data, labels, noise_type):
        network = self.__create_neural_network(data, labels, data, labels, is_custom_weights=True, custom_weights=weights)
        network.train(num_steps = 0, variable_storage_file_name = 'mod', verbose=False)
        raw_predictions = network.test_noisy_version('mod', noise_type=noise_type)[1]
        return raw_predictions
    
    def __create_neural_network(self, train_dataset, train_labels, valid_dataset, valid_labels,
                                is_custom_weights, custom_weights=None, is_composite_method=False, num_noise_types=None,
                                noise_dist=None):
        network = NeuralNetwork(train_dataset = train_dataset, 
                        train_labels = train_labels, 
                        valid_dataset = valid_dataset, 
                        valid_labels = valid_labels,
                        image_size = 28, 
                        num_labels = 10,
                        batch_size = 100,
                        num_hidden_layers = 1,
                        num_hidden_nodes = 1024,
                        optimizer_type = 'GradientDescent', 
                        optimizer_params ={'learning_rate': 0.5},
                        is_custom_weights = is_custom_weights,
                        custom_weights = custom_weights,
                        is_composite_method=is_composite_method,
                        num_noise_types=num_noise_types,
                        noise_dist=noise_dist)
        return network

    ############# Non-private functions. ###############
    
    def run_robust_optimizer(self, num_iterations):
        for i in xrange(num_iterations):
            self.iter_num = i
            self.train_info.append({})
            print "Starting Robust Optimizer Iteration: ", i

            new_result = self.__distributional_oracle(self.dist_over_noise)
            self.chosen_weights.append(new_result)

            if self.update_method == 'uniform_fixed':
                continue

            this_nu = self.update_method #eta = (log(m) / 2T)^(nu)

            for j in xrange(self.num_noise_types):
                unscaled_term = self.__custom_score(new_result, self.dataset2, self.labels2, self.noise_names[j])
                if unscaled_term < self.most_extreme_loss:
                    self.most_extreme_loss = unscaled_term
                self.dist_over_noise_scoring_sum[j] += ((unscaled_term / self.scale_bound) + 1.0)
                self.dist_over_noise[j] = np.exp(-1.0 * self.dist_over_noise_scoring_sum[j] * ((np.log(self.num_noise_types) / (2.0*num_iterations))**this_nu))
            self.dist_over_noise = self.__normalize_dist(self.dist_over_noise) 

            self.all_dist_over_noise.append(copy.deepcopy(self.dist_over_noise))
            print "Distribution over Noises: ", self.dist_over_noise
            self.train_info[self.iter_num]['Distribution over Noises'] = copy.deepcopy(self.dist_over_noise)
        return self.chosen_weights, self.all_dist_over_noise

    def check_test_loss(self, test_noise_types):
        #Checks individual loss of each solution within robust opt solution list, for each noise type.
        losses = {}
        for test_noise_type in test_noise_types:
            losses[test_noise_type] = []
            for i in xrange(len(self.chosen_weights)):
                losses[test_noise_type].append(self.__custom_score(self.chosen_weights[i], self.dataset3, self.labels3, test_noise_type))
        losses_means = {}
        for test_noise_type in test_noise_types:
            losses_means[test_noise_type] = np.asarray(losses[test_noise_type]).mean()

        self.test_info['Individual Losses'] = losses
        self.test_info['Mean of Individual Losses'] = losses_means
        return losses, losses_means

    def check_combined_loss(self, test_noise_types):
        #Uses all solutions together for ensemble classification, for each noise type.
        current_dataset = self.dataset3
        current_labels = self.labels3

        losses = {}
        for test_noise_type in test_noise_types:
            all_class_predictions = np.zeros((current_labels.shape[0], self.num_classes))
            for i in xrange(len(self.chosen_weights)):
                all_class_predictions += self.__custom_predict(self.chosen_weights[i], current_dataset, current_labels, test_noise_type)
            ensemble_softmax_vals = copy.deepcopy(all_class_predictions / len(self.chosen_weights))
            losses[test_noise_type] = self.__mean_cross_entropy(ensemble_softmax_vals, current_labels)
        self.test_info['Combined Losses'] = losses
        return losses

    def get_train_info(self):
        return self.train_info

    def get_test_info(self):
        return self.test_info