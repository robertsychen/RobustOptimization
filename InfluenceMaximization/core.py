import numpy as np
import copy
import itertools


class MasterGraph:
    def __init__(self, num_vertices, edge_list, probability):
        self.num_vertices = num_vertices
        self.edge_list = edge_list
        self.probability = probability

    def fetch_graph(self):
        '''
        Produces a subgraph (each edge included with self.probability) from master graph.
        '''
        this_edge_list = []
        rands = np.random.random(len(self.edge_list))
        for i in xrange(len(self.edge_list)):
            if rands[i] < self.probability:
                this_edge_list.append(self.edge_list[i])

        return Graph(self.num_vertices, this_edge_list)
    
    def fetch_many_graphs(self, num):
        '''
        Produces many subgraphs from master graph.
        '''
        graphs = []
        for i in xrange(num):
            graphs.append(self.fetch_graph())
        return graphs


class Graph:
    def __init__(self, num_vertices, edge_list):
        self.num_vertices = num_vertices
        self.edge_list = edge_list
        self.adj_list = {}

        for i in xrange(self.num_vertices):
            self.adj_list[i] = []
        for edge in self.edge_list:
            self.adj_list[edge[0]].append(edge[1])


class RobustOptimizer:
    def __init__(self, num_vertices, solution_size, graphs, score_scale=None):
        self.num_vertices = num_vertices
        self.solution_size = solution_size
        self.graphs = graphs
        self.num_graphs = len(graphs)
        self.dist_over_graphs = np.ones(self.num_graphs) / self.num_graphs
        self.all_dist_over_graphs = [self.dist_over_graphs]
        self.solutions = None
        self.score_scale = score_scale #used to scale influence values between 0 and 1
        if score_scale is None:
            self.score_scale = self.num_vertices
        self.max_evaluated_influence = 0.0 #to track whether score_scale value is sufficient; 
        #if self.max_evaluated_influence > self.score_scale, then need to increase score_scale parameter and then re-run simulation.
        self.robopt_solution_mins = []
        self.solutions_scores = None

    def compute_absolute_best_influence(self):
        '''
        Brute force finding set of nodes of size self.solution_size that maximizes influence.
        Should only be run when number of possible sets of size self.solution_size is sufficiently small.
        '''
        best_solution = None
        best_influence = 0.0
        all_combinations = [list(elem) for elem in list(itertools.combinations(range(self.num_vertices), self.solution_size))]
        for combo in all_combinations:
            worst_for_this_graph = 100.0
            for graph in self.graphs:
                this_influence = self.evaluate_influence(graph, combo)
                if this_influence < worst_for_this_graph:
                    worst_for_this_graph = this_influence
            if worst_for_this_graph > best_influence:
                best_influence = worst_for_this_graph
                best_solution = combo
        return best_influence, best_solution
        
    def evaluate_influence(self, graph, nodes):
        '''
        Given a set of nodes in a graph, computes how many nodes are reachable (influence).
        Also updates self.max_evaluated_influence is the current influence is greater.
        '''
        visited = np.zeros(self.num_vertices)
        stack = copy.deepcopy(nodes)
        while len(stack) > 0:
            this_index = stack.pop()
            visited[this_index] = 1
            for neighbor in graph.adj_list[this_index]:
                if visited[neighbor] == 0:
                    stack.append(neighbor)
        this_influence = visited.sum()
        if this_influence > self.max_evaluated_influence:
            self.max_evaluated_influence = this_influence
        return copy.deepcopy(this_influence)
    
    def find_reachable_nodes(self, graph, start):
        '''
        Returns indices of all nodes reachable in graph from given start vertex.
        '''
        reached = []
        stack = [start]
        while len(stack) > 0:
            this_index = stack.pop()
            reached.append(this_index)
            for neighbor in graph.adj_list[this_index]:
                if (neighbor not in reached) and (neighbor not in stack):
                    stack.append(neighbor)
        return reached
    
    def coverage_problem_builder(self):
        '''
        Formatting of reachable nodes from each node in all subgraphs.
        Formatted structure is exploited by coverage_greedy_solver.
        '''
        coverage_dict = {}
        for i in xrange(self.num_vertices):
            coverage_dict[i] = []
        for j in xrange(self.num_graphs):
            for vertex1 in self.graphs[j].adj_list.keys():
                reachable_nodes = self.find_reachable_nodes(self.graphs[j], vertex1)
                for vertex2 in reachable_nodes:
                    coverage_dict[vertex1].append([j,vertex2])
        return coverage_dict
    
    def baseline_coverage_problem_builder(self, graph):
        '''
        Formatting of reachable nodes from each node in graph.
        Formatting made to emulate coverage_problem_builder.
        '''
        coverage_dict = {}
        for i in xrange(self.num_vertices):
            coverage_dict[i] = []
        for vertex1 in graph.adj_list.keys():
            reachable_nodes = self.find_reachable_nodes(graph, vertex1)
            for vertex2 in reachable_nodes:
                coverage_dict[vertex1].append([0,vertex2]) 
                #^in order to share coverage_greedy_solver with coverage_problem_builder
        return coverage_dict
    
    def coverage_greedy_solver(self, orig_coverage_dict, is_weighted):
        '''
        Runs the (1-1/e) approximation greedy algorithm.
        If is_weighted == True, runs version for combining subgraphs, weighted by given distribution.
        If is_weighted == False, runs standard version used for single graphs.
        '''
        coverage_dict = copy.deepcopy(orig_coverage_dict)
        this_solution = []
        
        if is_weighted: #Robust optimizer and two of the baselines use this.
            weighting = copy.deepcopy(self.dist_over_graphs)
        else: #Individual-based baseline uses this.
            weighting = np.ones(self.num_graphs)

        #Populate total marginal weight adding each vertex would contribute.
        weight_left = np.zeros(self.num_vertices)
        for i in xrange(self.num_vertices):
            for pair in coverage_dict[i]:
                weight_left[i] += weighting[pair[0]]
                
        #Run greedy algorithm.
        for n in xrange(self.solution_size):
            if weight_left.max() == 0.0:
                break
            selected_node = np.argmax(weight_left)
            this_solution.append(selected_node)
            pairs_to_delete = copy.deepcopy(coverage_dict[selected_node])
            weight_left[selected_node] = 0.0
            del coverage_dict[selected_node]
            
            #Remove elements from remaining nodes in the coverage dictionary.
            for vertex in coverage_dict.keys():
                for pair in pairs_to_delete:
                    if pair in coverage_dict[vertex]:
                        coverage_dict[vertex].remove(pair)
                        weight_left[vertex] -= weighting[pair[0]]
        return this_solution
    
    def average_indiv_influence(self, solutions_scores):
        return solutions_scores.mean(axis=0).min()
        
    def run_robust_opt(self, num_iters, nu, dist_update_type='standard', is_printing_dist=False, distance_from_uniform=None):
        '''
        num_iters: how many simulations to run
        nu: parameter for MWU where eta = (log(m) / 2T)^(nu); can be anything if dist_update_type == 'fixed'
        dist_update_type: 'fixed' = keep uniform distribution over subgraphs (one of the 3 baselines)
            'standard' = standard robust optimization procedure
            'random' = random distribution per iteration over subgraphs with L1 dist. from uniform similar to robust opt (one of the 3 baselines)
        is_printing_dist: whether to print progress of dist. over subgraphs
        distance_from_uniform: L1 distances from uniform from previous robust opt run (only required when dist_update_type == 'random')
        Returns the average bottleneck influence across iterations and the solutions sets from all iterations.
        '''

        #Make sure starting fresh in case previous runs occurred with this class.
        self.dist_over_graphs = np.ones(self.num_graphs) / self.num_graphs
        self.all_dist_over_graphs = [self.dist_over_graphs]
        self.solutions = None
        self.max_evaluated_influence = 0.0
        self.robopt_solution_mins = []
        self.solutions_scores = None

        solutions = []
        solutions_scores = np.zeros((num_iters,self.num_graphs))
        solutions_scores_sums = np.zeros(self.num_graphs)
        #Set up coverage problem.
        coverage_dict = self.coverage_problem_builder()
        
        for i in xrange(num_iters):
            print i
            #Obtain new solution.
            this_solution = self.coverage_greedy_solver(coverage_dict, is_weighted=True)
            solutions.append(this_solution)
            
            if dist_update_type == 'fixed':
                for k in xrange(self.num_graphs):
                    solutions_scores[i,k] = self.evaluate_influence(self.graphs[k], this_solution) / self.score_scale #score scaled between 0 and 1

                #no reason to run mutitple iterations when input is constant
                for j in xrange(1,num_iters):
                    solutions_scores[j,:] = solutions_scores[0,:]
                break
            elif dist_update_type == 'standard':
                #Update distribution over graphs.
                unnormalized_dist = np.zeros(self.num_graphs)
                for k in xrange(self.num_graphs):
                    solutions_scores[i,k] = self.evaluate_influence(self.graphs[k], this_solution) / self.score_scale #score scaled between 0 and 1
                    solutions_scores_sums[k] += solutions_scores[i,k]
                    unnormalized_dist[k] = np.exp(-1.0 * solutions_scores_sums[k] * ((np.log(self.num_graphs) / (2.0*num_iters))**nu))
                self.dist_over_graphs = copy.deepcopy(unnormalized_dist) / unnormalized_dist.sum()
                self.all_dist_over_graphs.append(copy.deepcopy(self.dist_over_graphs))
                if is_printing_dist:
                    print self.dist_over_graphs
                    print 'Min: ', solutions_scores[i,:].min() * self.score_scale
                    print 'Max: ', solutions_scores[i,:].max() * self.score_scale
                self.robopt_solution_mins.append(solutions_scores[i,:].min() * self.score_scale)
            elif dist_update_type == 'random':
                for k in xrange(self.num_graphs):
                    solutions_scores[i,k] = self.evaluate_influence(self.graphs[k], this_solution) / self.score_scale #score scaled between 0 and 1
                assert(len(distance_from_uniform) == num_iters)
                unnormalized_dist = np.clip(((np.ones(self.num_graphs) / self.num_graphs) + ((np.random.random(self.num_graphs) - 0.5) * 2.0 * distance_from_uniform[i])),0,1)
                self.dist_over_graphs = copy.deepcopy(unnormalized_dist) / unnormalized_dist.sum()
                self.all_dist_over_graphs.append(copy.deepcopy(self.dist_over_graphs))
                if is_printing_dist:
                    print self.dist_over_graphs
                    print 'Min: ', solutions_scores[i,:].min() * self.score_scale
                    print 'Max: ', solutions_scores[i,:].max() * self.score_scale
            else:
                raise ValueError('dist_update_type not supported.')
            
        self.solutions = copy.deepcopy(solutions)
        self.solutions_scores = copy.deepcopy(solutions_scores) * self.score_scale
        
        #Evaluate performance.
        bottleneck_influence = self.average_indiv_influence(solutions_scores)
        print bottleneck_influence * self.score_scale
        
        return solutions, (bottleneck_influence*self.score_scale)
    
    def run_baseline(self, num_iters):
        '''
        Runs simulations using greedy fits from individual subgraphs as the solution set. (1 of the 3 baseline methods).
        Returns the average bottleneck influence across iterations and the solutions sets from all iterations.
        '''
        actual_num_iters = min(num_iters, self.num_graphs)

        #Make sure starting fresh in case previous runs occurred with this class.
        self.dist_over_graphs = np.ones(self.num_graphs) / self.num_graphs
        self.all_dist_over_graphs = [self.dist_over_graphs]
        self.solutions = None
        self.max_evaluated_influence = 0.0
        self.solutions_scores = None
        
        #use greedy fits for the individual graphs 
        solutions = []
        solutions_scores = np.zeros((actual_num_iters,self.num_graphs))
        for i in xrange(actual_num_iters):
            this_graph = self.graphs[i % self.num_graphs]
            this_coverage_setup = self.baseline_coverage_problem_builder(this_graph)
            this_solution = self.coverage_greedy_solver(this_coverage_setup, is_weighted=False)
        
            for k in xrange(self.num_graphs):
                solutions_scores[i,k] = self.evaluate_influence(self.graphs[k], this_solution) / self.score_scale #score scaled between 0 and 1

            solutions.append(this_solution)
        
        #Evaluate performance.
        bottleneck_influence = self.average_indiv_influence(solutions_scores)
        print bottleneck_influence * self.score_scale
        
        return solutions, (bottleneck_influence*self.score_scale)



