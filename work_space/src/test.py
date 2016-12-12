from multiprocessing import Process
import sys, os, math
from ctypes import cdll
sys.path.insert(0, os.getenv('lib'))
import util, init, solver, reducer

if __name__ == "__main__":  
    
    # config_file is the path to the configuration file [path-to]/data/input/configs_nu.txt, which you pass as argument when you run this script
    config_file         = util.getCommandLineArgs ()
    
    # 'M' is a NetworkX directed graph object, with an edge attribute named 'sign' that is either 1 (promotional) or -1 (inhibitory)
    # if you're not interested in loading that network you can just set M = nil, to free it from memory (equivelantly you can set 'network_file' parameter to nothing in your configs_nu.txt file
    # 'configs' is a dictionary holding all the paramters in the config_nu.txt file. You print this dictionary to see the name of the keys and the corresponding values
    M, configs          = init.initialize_master (config_file, 0) 
    print ("\nI loaded a network with "+str(len(M.nodes())) +" nodes (genes) " + str(len(M.edges())) +" edges\nRefer to Networkx documentation on how to look into or manipulate this network (adding/deleting nodes/edges etc)")
    
    # use this to call the solver
    knapsack_solver     = cdll.LoadLibrary(configs['KP_solver_binary'])
    
    # configs['PT_pairs_dict'][1][0] is a percentage of nodes, get the number of nodes corresponding to that percentage (if it's 100, then 'pressure' = total number of nodes)
    pressure            = math.ceil ((float(configs['PT_pairs_dict'][1][0])/100.0)*len(M.nodes())) 
    
    # configs['PT_pairs_dict'][1][1] is a percentage of edges (how many Oracle-contradicting edges to tolerate, after reduction this becomes the knapsack capacity)
    tolerance           = configs['PT_pairs_dict'][1][1] 
    
    # reverse_reduction(..) generates  random oracle advice, which results in having an NEP instance, that instance is reduced to a knapsack instance
    # ignore this parameters: configs['advice_upon'], configs['biased'], configs['BD_criteria']
    kp_instances = reducer.reverse_reduction(M, pressure, tolerance, configs['sampling_rounds'], configs['advice_upon'], configs['biased'], configs['BD_criteria']) # 
    
    i=1
    print ("\nI'm going to solve "+str(configs['sampling_rounds'])+" knapsack instances\n\nHere we go:")
    for kp in kp_instances: # in each iteration, a new knapsack instances is 'yeilded' by reverse_reduction function
        #solve the instance
        a_result = solver.solve_knapsack (kp, knapsack_solver)
        print ("instance # "+str(i).ljust(15," "), end='')
        i+=1
    
        # the solver returns the following as a list:
        # 0		TOTAL_Bin:		total value of objects inside the knapsack, 
        # 1		TOTAL_Din:		total weight of objects inside the knapsack 
        # 2		TOTAL_Bout:		total value of objects outside knapsack 
        # 3		TOTAL_Dout:		total weight of object outside knapsack, 
        # 4		GENES_in: 		a list, each element in the list is a tuple of three elements: node(gene) ID, its value(benefit), its weight(damage)
        # 5		GENES_out:		a list of tuples also, as above, with genes that are outside the knapsack	
        # 6		green_genes:	a list of tuples also, as above, with genes that have values greater than zero but zero weights (hence they are automatically inside the knaspack and not included in the optimization)	 
        # 7		red_genes:		a list of tuples also, as above, with genes that have weights greater than zero but zero values (hence they are automatically outside the knaspack and not included in the optimization)
        # 8		grey_genes:		a list of tuples also, as above, with genes that have greater than zero values and weights (these are the nodes that were optimized over to see which should be in and which should be out) 
        # 9		coresize: 		ignore
        # 10	execution_time: in seconds
        
        if len (a_result)>0:
            print ("knapsack value = "+str(a_result[0]).ljust(15,' ')+"knapsack weight "+str(a_result[1]))
            Gs, Bs, Ds, Xs = [],[],[],[]
            # -------------------------------------------------------------------------------------------------
            GENES_in, GENES_out, coresize, execution_time = a_result[4], a_result[5], a_result[9], a_result[10]
            # -------------------------------------------------------------------------------------------------
            for g in GENES_in: # notice that green_genes is a subset of GENES_in
                Gs.append(g[0]+'$'+str(M.in_degree(g[0]))+'$'+str(M.out_degree(g[0])))
                Bs.append(g[1])
                Ds.append(g[2])
                Xs.append(1)
            for g in GENES_out: # notice that red_genes is a subset of GENES_out
                Gs.append(g[0]+'$'+str(M.in_degree(g[0]))+'$'+str(M.out_degree(g[0])))
                Bs.append(g[1])
                Ds.append(g[2])
                Xs.append(0)
        else:
            print ("void")
            # you can add code here to handle these vectors: Gs, Bs, Ds, Xs, which are, respectively, the genes, their corresponding benefits, their corresponding weights, and the solution vector (a binary 0/1 sequence, 0 = outside knapsack, 1=inside knapsack)
    print ("Done")
    
    