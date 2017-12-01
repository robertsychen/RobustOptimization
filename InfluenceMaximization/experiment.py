import numpy as np
import pickle
import time
import copy

from core import Graph
from core import MasterGraph
from core import RobustOptimizer


def get_distance_from_uniforms(dists):
    '''
    Get L-1 distances from uniform distribution for several distributions. Used for the random distribution-based baseline.
    '''
    distance_from_uniforms = []
    for dist in dists:
        distance_from_uniforms.append(np.absolute((np.ones(dist.shape[0]) / dist.shape[0]) - dist).mean())
    return distance_from_uniforms[1:]


def social_influence_experiment(dataset,
                                num_vertices,
                                probability,
                                num_subgraphs,
                                solution_size,
                                score_scale,
                                num_iters,
                                num_runs=1,
                                pkl_name='test',
                                nu=0.5,
                                verbose=False,
                                existing_old_graphs=None):
    '''
    dataset: which master graph to use -- 'wiki', 'complete'
    num_vertices: num vertices in that master graph: 7115 for 'wiki', 100 for 'complete'
    probability: how likely each edge is included in any given subgraph of the master graph
    num_subgraphs: num subgraphs (aka m in the paper)
    solution_size: num vertices in any seed set solution for the max influence problem
    score_scale: used to scale influence values between 0 and 1 (in line with theory) for part of the experiment. Set score_scale to be at least
        greater than highest expected/possible influence value for any seed set, but not too much larger (within factor of 2 is good for example)
    num_iters: number of seed sets in a solution set for the algorithm (aka T in the paper)
    num_runs: num trials to run & average results over
    pkl_name: file name to dump resulting subgraphs, performance statistics from experiment, etc. If None, results will be printed but not saved.
    nu: value used in MWU where eta = (log(m) / 2T)^(nu)
    verbose: whether to print progress of distribution updates across iterations in each run
    existing_old_graphs: if None, generates new random subgraphs to use. Else, uses provided set of subgraphs to run experiment with.
    '''
    time1 = time.time()

    edge_list = None
    if dataset == 'wiki':
        edge_list = pickle.load(open("wiki_edge_list.pkl", "rb"))
    elif dataset == 'complete':
        edge_list = []
        for i in xrange(num_vertices):
            for j in xrange(num_vertices):
                if i != j:
                    edge_list.append([i,j])
    else:
        raise ValueError('Dataset not supported.')
    
    baseline1_scores = np.zeros(num_runs) #individual consituent solutions
    robopt_scores = np.zeros(num_runs)
    baseline2_scores = np.zeros(num_runs) #random perturbation-based
    baseline3_scores = np.zeros(num_runs) #pure uniform dist
    robopt_indiv_mins = []
    overall_max_robopt_influence = 0.0
    all_graphs = []
    all_solutions_scores = []
    all_solutions_scoresbase = []
    
    for i in xrange(num_runs):
        print "Running run ", i
        graphs = None
        if not existing_old_graphs:
            mastergraph = MasterGraph(num_vertices, edge_list, probability=probability)
            graphs = mastergraph.fetch_many_graphs(num_subgraphs)
        else:
            graphs = existing_old_graphs[i]
        robopt = RobustOptimizer(num_vertices=num_vertices,
                             solution_size=solution_size,
                             graphs=graphs,
                             score_scale=score_scale)
        
        baseline1_scores[i] = robopt.run_baseline(num_iters=num_iters)[1]
        baseline3_scores[i] = robopt.run_robust_opt(num_iters=num_iters, nu=nu, dist_update_type='fixed',
                                                 is_printing_dist=verbose)[1]
        robopt_scores[i] = robopt.run_robust_opt(num_iters=num_iters, nu=nu, dist_update_type='standard',
                                                 is_printing_dist=verbose)[1]
        robopt_indiv_mins.append(max(robopt.robopt_solution_mins))
        all_solutions_scores.append(copy.deepcopy(robopt.solutions_scores))
        if robopt.max_evaluated_influence > overall_max_robopt_influence:
            overall_max_robopt_influence = robopt.max_evaluated_influence
        if pkl_name:
            pickle.dump(robopt, open(pkl_name + str(i) + ".pkl", "wb"))
        dist_unifs = get_distance_from_uniforms(robopt.all_dist_over_graphs)
        baseline2_scores[i] = robopt.run_robust_opt(num_iters=num_iters, nu=nu, dist_update_type='random',
                                                    is_printing_dist=verbose, distance_from_uniform=dist_unifs)[1]
        all_solutions_scoresbase.append(copy.deepcopy(robopt.solutions_scores))
        all_graphs.append(robopt.graphs)

    results = {}
    results['Baseline1 Scores'] = baseline1_scores
    results['Robopt Scores'] = robopt_scores
    results['Baseline2 Scores'] = baseline2_scores
    results['Baseline3 Scores'] = baseline3_scores
    results['Baseline1 Mean'] = baseline1_scores.mean()
    results['Baseline1 SD'] = baseline1_scores.std()
    results['Robopt Mean'] = robopt_scores.mean()
    results['Robopt SD'] = robopt_scores.std()
    results['Baseline2 Mean'] = baseline2_scores.mean()
    results['Baseline2 SD'] = baseline2_scores.std()
    results['Baseline3 Mean'] = baseline3_scores.mean()
    results['Baseline3 SD'] = baseline3_scores.std()
    results['Robopt - Baseline1 Mean'] = (robopt_scores-baseline1_scores).mean()
    results['Robopt - Baseline1 SD'] = (robopt_scores-baseline1_scores).std()
    results['Robopt - Baseline2 Mean'] = (robopt_scores-baseline2_scores).mean()
    results['Robopt - Baseline2 SD'] = (robopt_scores-baseline2_scores).std()
    results['Robopt - Baseline3 Mean'] = (robopt_scores-baseline3_scores).mean()
    results['Robopt - Baseline3 SD'] = (robopt_scores-baseline3_scores).std()
    results['Max Influence Evaluated'] = overall_max_robopt_influence
    results['Robopt Max Indiv Mins'] = robopt_indiv_mins
    results['Indiv Best to Robopt Criterion Ratio'] = np.divide(robopt_indiv_mins, robopt_scores)
    results['Time'] = time.time() - time1
    results['Solution Scores, Robopt'] = all_solutions_scores
    results['Solution Scores, Random'] = all_solutions_scoresbase
     
    for key, value in sorted(results.items()):
        print key, value

    pickle.dump(results, open(pkl_name + "masterresults.pkl", "wb"))
    pickle.dump(all_graphs, open(pkl_name + "mastergraphs.pkl", "wb"))
    return results, all_graphs


def compute_absolute_best_wrapper(dataset,
                                num_vertices,
                                probability,
                                num_subgraphs,
                                solution_size,
                                existing_old_graph=None):
    '''
    Does setup for use of robopt.compute_absolute_best_influence(), then calls it 
    to get best seed set/influence value possible for given set of subgraphs, using brute force.
    dataset: which master graph to use -- 'wiki', 'complete'
    num_vertices: num vertices in that master graph: 7115 for 'wiki', 100 for 'complete'
    probability: how likely each edge is included in any given subgraph of the master graph
    num_subgraphs: num subgraphs (aka m in the paper)
    solution_size: num vertices in any seed set solution for the max influence problem
    existing_old_graphs: if None, generates new random subgraphs to use. Else, uses provided set of subgraphs to run experiment with.
    '''
    time1 = time.time()

    edge_list = None
    if dataset == 'wiki':
        edge_list = pickle.load(open("wiki_edge_list.pkl", "rb"))
    elif dataset == 'complete':
        edge_list = []
        for i in xrange(num_vertices):
            for j in xrange(num_vertices):
                if i != j:
                    edge_list.append([i,j])
    else:
        raise ValueError('Dataset not supported.')

    graphs = None
    if not existing_old_graph:
        mastergraph = MasterGraph(num_vertices, edge_list, probability=probability)
        graphs = mastergraph.fetch_many_graphs(num_subgraphs)
    else:
        graphs = existing_old_graph
    robopt = RobustOptimizer(num_vertices=num_vertices,
                         solution_size=solution_size,
                         graphs=graphs,
                         score_scale=None)
        
    result = robopt.compute_absolute_best_influence()
    print time.time() - time1
    return result