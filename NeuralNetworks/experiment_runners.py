import numpy as np
import copy
import time
import pickle
from scipy import stats

from neural_network import NeuralNetwork
from robust_optimizer import RobustOptimizer
from mnist import get_mnist_data

def obtain_mnist_data():
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_mnist_data()
    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels

def mean_cross_entropy(dists1, dists2):
    assert(dists1.shape == dists2.shape)
    num = dists1.shape[0]
    total = 0.0
    for i in xrange(num):
        total += cross_entropy(dists1[i,:], dists2[i,:])
    return total / num

def cross_entropy(dist1, dist2):
    #for our use case, dist1 is the prediction, dist2 is the true labels
    result = 0.0
    small_float = 1.0e-307
    for i in xrange(dist1.shape[0]):
        result += (dist2[i] * np.log(dist1[i] + small_float))
    return result

def run_general_noise_experiment(train_noises, 
                                test_noises, 
                                num_trains,
                                num_components,
                                combining_method='individual_avg', 
                                num_classes=10, 
                                save_name='test'):
    '''
    User-friendly function to run general experiments on test loss under different training and test noises.
    Tests loss for every ordered pair of (train_noise, test_noise) given as input.
    Prints key statistics; also stores more detailed data in a Python dictionary called results, which is returned 
    by the function and also saved as a .pkl file.
    Used to construct the noise behavior tables in the paper.

    Parameter notes: 
    num_trains is number of trials; num_components is value of T
    combining_method is one of {individual_avg, softmax}. They correspond to Individual Bottleneck and Ensemble Bottleneck, respectively.
    '''
    #Including 'even_split' in the train_noises list implements the Even Split Baseline that split among all other train noises equally. 
    #If 'even_split' is included, it must be the last noise listed in the train_noises list.

    results = {}
    start_time = time.time()

    [train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels] = obtain_mnist_data()
    
    if 'even_split' in train_noises:
        assert(train_noises[-1] == 'even_split')
    
    all_losses = {}
    for train_noise in train_noises:
        all_losses[train_noise] = {}
        for test_noise in test_noises:
            all_losses[train_noise][test_noise] = []

    for train_noise in train_noises:
        print "Train Noise Type: ", train_noise
        for i in xrange(num_trains):
            print "Train number: ", i
            all_class_predictions = {}
            for test_noise in test_noises:
                if combining_method == 'softmax':
                    all_class_predictions[test_noise] = np.zeros((test_labels.shape[0], num_classes))
                elif combining_method == 'individual_avg':
                    all_class_predictions[test_noise] = np.zeros(num_components)
            for j in xrange(num_components):
                print "Component: ", j
                network = NeuralNetwork(train_dataset = train_dataset, 
                                        train_labels = train_labels, 
                                        valid_dataset = test_dataset, 
                                        valid_labels = test_labels,
                                        image_size = 28, 
                                        num_labels = 10,
                                        batch_size = 100,
                                        num_hidden_layers = 1,
                                        num_hidden_nodes = 1024,
                                        optimizer_type = 'GradientDescent', 
                                        optimizer_params ={'learning_rate': 0.5})
                if train_noise != 'even_split':
                    network.train(num_steps = 500, variable_storage_file_name = 'mod',
                                      verbose=False, noise_type=train_noise)
                else:
                    all_train_noises = train_noises[0:(len(train_noises)-1)]
                    this_train_noise = all_train_noises[j % len(all_train_noises)]
                    network.train(num_steps = 500, variable_storage_file_name = 'mod',
                                      verbose=False, noise_type=this_train_noise)
                
                if combining_method == 'softmax':
                    for test_noise in test_noises:
                        _, raw_predictions = network.test_noisy_version('mod', noise_type=test_noise)
                        all_class_predictions[test_noise] += raw_predictions
                elif combining_method == 'individual_avg':
                    for test_noise in test_noises:
                        this_loss, _ = network.test_noisy_version('mod', noise_type=test_noise)
                        all_class_predictions[test_noise][j] = this_loss
                    
            
            if combining_method == 'softmax':
                for test_noise in test_noises:
                    all_losses[train_noise][test_noise].append(mean_cross_entropy(all_class_predictions[test_noise] / num_components, test_labels))
            elif combining_method == 'individual_avg':
                for test_noise in test_noises:
                    all_losses[train_noise][test_noise].append(all_class_predictions[test_noise].mean())
    results['Raw'] = all_losses

    results['Table'] = {}
    for train_noise in train_noises:
        for test_noise in test_noises:
            print train_noise, test_noise
            print np.asarray(all_losses[train_noise][test_noise]).mean(), (np.asarray(all_losses[train_noise][test_noise]).std() / np.sqrt(num_trains))
            results['Table'][train_noise + ' ' + test_noise] = [np.asarray(all_losses[train_noise][test_noise]).mean(), (np.asarray(all_losses[train_noise][test_noise]).std() / np.sqrt(num_trains))]
    
    results['Bottleneck Summary'] = {}
    print "Max-Min Info:"
    max_min_info = {}
    for train_noise in train_noises:
        max_min_info[train_noise] = []
        for i in xrange(num_trains):
            bottleneck_loss = None
            bottleneck_noise = None
            for test_noise in test_noises:
                if bottleneck_loss is None:
                    bottleneck_loss = all_losses[train_noise][test_noise][i]
                    bottleneck_noise = test_noise
                elif bottleneck_loss > all_losses[train_noise][test_noise][i]:
                    bottleneck_loss = all_losses[train_noise][test_noise][i]
                    bottleneck_noise = test_noise
            max_min_info[train_noise].append((bottleneck_loss, bottleneck_noise))
        print train_noise
        print "Bottleneck Mean: ", np.asarray([elem[0] for elem in max_min_info[train_noise]]).mean()
        print "SD of Bottleneck Mean: ", np.asarray([elem[0] for elem in max_min_info[train_noise]]).std() / np.sqrt(num_trains)
        results['Bottleneck Summary'][train_noise] = [np.asarray([elem[0] for elem in max_min_info[train_noise]]).mean(), np.asarray([elem[0] for elem in max_min_info[train_noise]]).std() / np.sqrt(num_trains)]
    results['Bottleneck Full'] = max_min_info

    results['Total Time'] = time.time() - start_time
    pickle.dump(results, open(save_name+".pkl", "wb"))
    return results


def run_robust_opt_experiment(num_runs, 
                            num_component_solutions,
                            this_noise_names, 
                            save_name,
                            this_oracle_method, 
                            this_update_method, 
                            training_num_iters=500, 
                            scale_bound=3.0,
                            initial_dist_over_noise=None):
    '''
    User-friendly function to run experiments using robust optimization algorithms. Computes variety of statistics 
    (for the negated mean loss) for the experiments. 
    Prints key statistics; also stores more detailed data in a Python dictionary called results, which is returned 
    by the function and also saved as a .pkl file.

    Parameter notes: 
    num_runs: # overall simulations to do and average statistics for
    num_component_solutions: value of T
    this_noise_names: list of all noise types in this noise set
    save_name: filename to create to save results
    this_oracle_method: can be either {hybrid, composite}. Corresponds to Hybrid Oracle vs. Composite Oracle.
    this_update_method: can be either {'uniform_fixed', nu float value}. Use former for uniform dist baseline & latter otherwise.
    training_num_iters: number of mini-batches to train neural network
    scale_bound: used by robust optimization for updates. Needed to scale loss values between 0 and 1. Choose s.t. greater than max
        of absolute value of loss ever generated by NeuralNetwork.test_noisy_version().
    initial_dist_over_noise: optional setting of initial distribution over noises, otherwise defaults to uniform distribution
    '''

    [train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels] = obtain_mnist_data()

    results = {}
    start_time = time.time()

    all_info = []
    for i in xrange(num_runs):
        time1 = time.time()
        robustopt = RobustOptimizer(dataset1=train_dataset,
                                    labels1=train_labels,
                                    dataset2=valid_dataset,
                                    labels2=valid_labels,
                                    dataset3=test_dataset,
                                    labels3=test_labels, 
                                    noise_names=this_noise_names,
                                    oracle_method=this_oracle_method,
                                    update_method=this_update_method,
                                    training_num_iters=training_num_iters,
                                    scale_bound=scale_bound,
                                    initial_dist_over_noise=initial_dist_over_noise)
        robustopt.run_robust_optimizer(num_component_solutions)
        print "Most Extreme Loss: ", robustopt.most_extreme_loss
        print "Train Time: ", time.time() - time1
        time1 = time.time()
        res, _ = robustopt.check_test_loss(this_noise_names)
        print "Individual Results"
        for key in res:
            print key
            print res[key]
            print np.asarray(res[key]).mean()
        res2 = robustopt.check_combined_loss(this_noise_names)
        print "Ensemble Results"
        print "Test Time: ", time.time() - time1
        print res2
        info = {'train': robustopt.get_train_info(), 'test': robustopt.get_test_info()}
        all_info.append(info)
    results['Raw'] = all_info
    
    #Reformatting data.
    results['By Test Noise Summary, Combined'] = {}
    start_losses1 = [elem['test']['Combined Losses'] for elem in all_info]
    inverted_losses1 = {}
    for name in this_noise_names:
        inverted_losses1[name] = []
    for i in xrange(len(start_losses1)):
        for key in start_losses1[i]:
            inverted_losses1[key].append(start_losses1[i][key])
    for key in inverted_losses1:
        print key
        print np.asarray(inverted_losses1[key]).mean(), np.asarray(inverted_losses1[key]).std() / np.sqrt(num_runs)
        results['By Test Noise Summary, Combined'][key] = [np.asarray(inverted_losses1[key]).mean(), np.asarray(inverted_losses1[key]).std() / np.sqrt(num_runs)]

    #Reformatting data.
    results['By Test Noise Summary, Individual'] = {}
    start_losses2 = [elem['test']['Mean of Individual Losses'] for elem in all_info]
    inverted_losses2 = {}
    inverted_losses2[name] = []
    for name in this_noise_names:
        inverted_losses2[name] = []
    for i in xrange(len(start_losses2)):
        for key in start_losses2[i]:
            inverted_losses2[key].append(start_losses2[i][key])
    for key in inverted_losses2:
        print key
        print np.asarray(inverted_losses2[key]).mean(), np.asarray(inverted_losses2[key]).std() / np.sqrt(num_runs)
        results['By Test Noise Summary, Individual'][key] = [np.asarray(inverted_losses2[key]).mean(), np.asarray(inverted_losses2[key]).std() / np.sqrt(num_runs)]
        
    #Softmax Ensemble Loss on Test Data.
    print 'Cumulative Max-Min Info for Combined Loss:'
    maxmin_info = []
    for i in xrange(len(start_losses1)):
        min_loss = None
        corr_noise = None
        for noise_key in start_losses1[i]:
            if not min_loss:
                min_loss = start_losses1[i][noise_key]
                corr_noise = noise_key
            elif start_losses1[i][noise_key] < min_loss:
                min_loss = start_losses1[i][noise_key]
                corr_noise = noise_key
        maxmin_info.append([min_loss, corr_noise])
        print min_loss, corr_noise
    print "Mean Bottleneck:", np.asarray([elem[0] for elem in maxmin_info]).mean()
    print "SD of Mean Bottleneck:", np.asarray([elem[0] for elem in maxmin_info]).std() / np.sqrt(len(start_losses1))
    results['Full Bottleneck Data, Combined'] = [elem[0] for elem in maxmin_info]
    results['Mean Bottleneck, Combined'] = np.asarray([elem[0] for elem in maxmin_info]).mean()
    results['Mean Bottleneck SD, Combined'] = np.asarray([elem[0] for elem in maxmin_info]).std() / np.sqrt(len(start_losses1))

    #Individual Bottleneck Loss on Test Data.
    print 'Cumulative Max-Min Info for Individual Loss:'
    maxmin_info = []
    for i in xrange(len(start_losses2)):
        min_loss = None
        corr_noise = None
        for noise_key in start_losses2[i]:
            if not min_loss:
                min_loss = start_losses2[i][noise_key]
                corr_noise = noise_key
            elif start_losses2[i][noise_key] < min_loss:
                min_loss = start_losses2[i][noise_key]
                corr_noise = noise_key
        maxmin_info.append([min_loss, corr_noise])
        print min_loss, corr_noise
    print "Mean Bottleneck:", np.asarray([elem[0] for elem in maxmin_info]).mean()
    print "SD of Mean Bottleneck:", np.asarray([elem[0] for elem in maxmin_info]).std() / np.sqrt(len(start_losses2))
    results['Full Bottleneck Data, Individual'] = [elem[0] for elem in maxmin_info]
    results['Mean Bottleneck, Individual'] = np.asarray([elem[0] for elem in maxmin_info]).mean()
    results['Mean Bottleneck SD, Individual'] = np.asarray([elem[0] for elem in maxmin_info]).std() / np.sqrt(len(start_losses2))

    results['Total Time'] = time.time() - start_time
    pickle.dump(results, open(save_name+".pkl", "wb"))
        
    return results