#!/usr/local/bin/python3
import math, operator, os, random, sys
from ctypes import cdll
import multiprocessing as mp

import networkx as nx

sys.path.insert(0, os.getenv('lib'))
import util, init, solver, reducer
from work_space.src import output, plot_nets


class Net:
    def __init__(self, net, id):
        self.fitness = 0    #aim to max
        self.fitness_parts = [0]*2   #effective total benefits
        self.net = net.copy()
        self.rebirths = 0
        self.id = id

    def copy(self):
        copy = Net(self.net, self.id)
        copy.fitness = self.fitness
        copy.fitness_parts = self.fitness_parts
        copy.rebirths = self.rebirths
        return copy


# GRAPH FN'S
def add_edge(net):
    if (len(net.nodes()) < 2):
        print("ERROR add_edge(): cannot add edge to net with < 2 nodes")
        return -1
    node1 = random.sample(net.nodes(), 1)
    node2 = random.sample(net.nodes(), 1)
    while (node1 == node2):                             #poss infinite loop?
        node2 = random.sample(net.nodes(),1)
    sign = random.randint(0, 1)
    if (sign == 0):     sign = -1
    #if (node1 not in net.nodes() or node2 not in net.nodes()) :
    net.add_edge(node1[0], node2[0], sign=sign)

def connect_components(net):
    #finds connected components and connects 'em
    components = list(nx.weakly_connected_component_subgraphs(net))
    while  (len(components) != 1):
        for i in range(len(components)-1):
            #might be faster method than SystemRandom
            node1 = random.SystemRandom().sample(components[i].nodes(), 1)
            node2 = random.SystemRandom().sample(components[i+1].nodes(), 1)
            if (node1 == node2): print("WARNING connect_components(): somehow same node is picked btwn two diff components.")
            sign = random.randint(0, 1)
            if (sign == 0):     sign = -1
            net.add_edge(node1[0], node2[0], sign=sign)
        components = list(nx.weakly_connected_component_subgraphs(net))

# EVO FN'S
def breed(population, num_survive, pop_size, cross_thresh, mutation_freq):
    #duplicate, cross, or random
    population.sort(key=operator.attrgetter('fitness'))
    population.reverse()
    #print ("Least fit=" + str(population[0].fitness) + " vs most fit=" + str(population[len(population)-1].fitness))
    #aim to MAX fitness

    for p in range(0, pop_size-num_survive):
        #duplicate
        population[p].rebirths += 1

        repro_type = random.random()
        if (repro_type < cross_thresh): #sexual
            rand1 = random.randint(0,num_survive-1)
            rand2 = random.randint(0, num_survive-1)
            while (rand1==rand2):
                rand2 = random.randint(0, num_survive-1)
            population[p].net = cross(population[rand1].net, population[rand2].net)

        else:   #asexual
            rand = random.randint(0,num_survive-1)           #check that gives right range
            population[p].net = population[rand].net.copy()

        for i in range(mutation_freq):
            mutate(population[p].net)


def cross(parent1, parent2):
    #TEST
    #later: change merge, diff sizes of two parents should be possible
    if ((len(parent1.nodes())) != (len(parent2.nodes()))): print("ERROR: two crossing parents don't have same # nodes!" + str((len(parent1.nodes()))) + str(len(parent2.nodes())))
    split = random.randint(0,len(parent1.nodes())-1)
    rand_nodes1 = random.SystemRandom().sample(parent1.nodes(), split)
    rand_subgraph1 = parent1.subgraph(rand_nodes1).copy()
    rand_nodes2 = random.SystemRandom().sample(parent2.nodes(), len(parent2.nodes())-split)
    rand_subgraph2 = parent2.subgraph(rand_nodes2).copy()

    #rand_nodes1 = random.SystemRandom().sample(parent1.nodes(), int((len(parent1.nodes())+1)/2))
    #rand_nodes2 = random.SystemRandom().sample(parent2.nodes(), int(len(parent2.nodes())/2))

    #print("\n\nrand nodes 1: " + str(rand_subgraph1.nodes()))
    #print("rand edges 1: " + str(rand_subgraph1.edges()))
    #print("\nrand nodes 2: " + str(rand_subgraph2.nodes()))
    #print("rand edges 2: " + str(rand_subgraph2.edges()))

    child = nx.disjoint_union(rand_subgraph1, rand_subgraph2)

    #print("\nchild nodes: " + str(child.nodes()))
    #print("child edges: " + str(child.edges()))

    components = list(nx.weakly_connected_component_subgraphs(child))
    if (len(components) > 20): print("WARNING cross(): child from " + str(len(components)) + " components.")
    #basically means nets aren't dense enough
    #maybe nature of taking random subsamples instead of random connected components/subgraphs

    connect_components(child)

    return child

def cross_bfs(parent1, parent2):
    #passes and returns nets

    if ((len(parent1.nodes())) != (len(parent2.nodes()))): print("WARNING: two crossing parents don't have same # nodes!" + str((len(parent1.nodes()))) + str(len(parent2.nodes())))
    size1 = random.randint(0,len(parent1.nodes())-1)
    size2 = len(parent1.nodes())-size1

    #diff btwn these two rands?
    node1 = random.SystemRandom().sample(parent1.nodes(),1)
    node2 = random.choice(parent2.nodes())
    bfs1 = nx.bfs_edges(parent1, node1)
    bfs2 = nx.bfs_edges(parent2, node2)

    component1 = bfs1[:size1]
    component2 = bfs2[:size2]

    child = nx.disjoint_union(component1, component2)
    if (len(child.nodes()) != len(parent1.nodes())): print("WARNING: resulting child is different size from parents.")
    if (len(list(nx.weakly_connected_component_subgraphs(child))) != 2): print("ERROR: child is not 2 components, but " + str(len(list(nx.weakly_connected_component_subgraphs(child)))) + " instead.")

    connect_components(child)

    return child



def cross_alt(parent1, parent2):
    #takes NET of both parents as inputs
    # "frags" are broken pieces of parents
    #parents = [parent1, parent2]
    frag1 = parent1.copy()
    frag2 = parent2.copy()
    frags = [frag1, frag2]
    child = frag1
    brokenList = [[]for i in range(2)]
    print("frag1 edges before: " + str(frag1.edges()))
    for p in range(2):
        node2 = node1 = random.choice(frags[p].nodes()) #btr node choice method, change else where too
        while(node2 == node1):
            node2 = random.choice(frags[p].nodes())
        frags[p].add_node("BROKEN")
        while (nx.has_path(frags[p], node1, node2)):
            path = list(nx.shortest_path(frags[p], node1, node2))  #shortest path or random path?
            print("PATH: " + str(path))
            success = False
            while (success == False):  #poss infinite loop
                path_index = random.randint(0,len(path)-1)   #might rm whole path, instead of single edge
                while (path[path_index] == "BROKEN" or path_index == len(path)-1): path_index = random.randint(0,len(path)-1)
                path_node1 = path[path_index]
                print("index1: " + str(path_index))
                path_index += 1
                success = True
                while (path[path_index] == "BROKEN"):
                    if (path_index+1 > len(path)):
                        success = False
                        break
                    path_index += 1

            path_node2 = path[path_index]
            print("index2: " + str(path_index))
            print("EDGE: " + str(path_node1) + ", " +  str(path_node2))
            frags[p].remove_edge(path_node1, path_node2)    #instead should rm random edge in path
            frags[p].add_edge(path_node1, "BROKEN")
            brokenList[p].append(path_node2)

    if (len(list(nx.weakly_connected_component_subgraphs(frag1))) != 2):
        print("Wrong # of components, found " + str(len(list(nx.weakly_connected_component_subgraphs(frag1)))))

    for p in range(2):
        for b in brokenList[p]:
            frags[p].add_edge("BROKEN", b)

    c = list(nx.weakly_connected_component_subgraphs(frags[0])) #HERE doesnt actually split
    frag1 = c[0]
    c = list(nx.weakly_connected_component_subgraphs(frags[1]))
    frag2 = c[0]
    print("frag1 edges after: " + str(frag1.edges()))
    fraggy = c[1]
    print("fraggy edges after: " + str(fraggy.edges()))
    print(frag1.edges("BROKEN"))

    while (frag1.edges("BROKEN")):       #how to write if edge exists btwn BROKEN and any other?
        if (frag2.edges("BROKEN")):
            print("HERE")


    return child
    #later: pick 2 nodes in each parent and divide graph in half, combine for child
    #normlz for graph size

def evolve_worker(worker_ID, founder_pop, num_workers, gens, num_survive, crossover_percent, mutation_freq, output_dir, pressure, tolerance, configs):
    #have to pass a lot more params for: breed, pressurize
    #diff recogz class Net list as Nets

    print("Worker " + str(worker_ID) + " starting.")

    knapsack_solver     = cdll.LoadLibrary(configs['KP_solver_binary'])
    #survive_fraction = float(configs['percent_survive'])/100
    grow_freq = float(configs['growth_frequency'])
    output_freq = float(configs['output_frequency'])
    avg_degree = int(configs['average_degree'])

    founder_size = len(founder_pop)
    output_dir += str(worker_ID)

    if (founder_size < num_survive):
        print("WARNING in evolve_worker(): more than founder population used for initial breeding.")
        #doesn't entirely break algo, so allowed to continue

    #build initial population by breeding founders
    '''
    pop_size = founder_size #redundant
    population = [Net(nx.DiGraph(),i) for i in range(pop_size)]
    for i in range(founder_size):
        population[i] = founder_pop[i]
    breed(population, founder_size, pop_size, crossover_percent, mutation_chance)
    #note: founder_size and pop_size must be diff for breed to do anything
    '''
    population = founder_pop
    pop_size = founder_size

    for g in range(gens):
        max_RGGR = max_ETB = 0
        min_RGGR = min_ETB = 1000000

        for p in range(pop_size):
            if (grow_freq != 0 and g % int(gens/grow_freq) == 0):
                grow(population[p].net, 10, avg_degree)  #net, startsize, avgdegree
                #choice of GROWTH FUNCTION, eventually dyn slows
                if (p == 0): console_report(population[0])

            population[p].fitness_parts = pressurize(population[p].net, pressure, tolerance, knapsack_solver)
            max_RGGR = max(max_RGGR, population[p].fitness_parts[0])
            max_ETB = max(max_ETB, population[p].fitness_parts[1])
            min_RGGR = min(min_RGGR, population[p].fitness_parts[0])
            min_ETB = min(min_ETB, population[p].fitness_parts[1])
            #print(population[p].fitness_parts)

        for p in range(pop_size):
            population[p].fitness = .5*(population[p].fitness_parts[0]-min_RGGR)/(max_RGGR-min_RGGR) + .5*(population[p].fitness_parts[1]-min_ETB)/(max_ETB-min_ETB)
            #might want to check that max_RGGR and max_ETB != 0 and throw warning if are (and divide by 1)

        breed(population, num_survive, pop_size, crossover_percent, mutation_freq)
        if (output_freq != 0 and g % int(1/output_freq) == 0):
            popn_info = [len(population[0].net.nodes()),max_RGGR, min_RGGR, max_ETB, min_ETB]
            output.to_csv(population, output_dir, popn_info)  # need to write output.worker_csv()

    population.sort(key=operator.attrgetter('fitness'))
    population.reverse()
    #as in breed(), check that sorts with MAX fitness first
    #print("In worker " + str(worker_ID) + ": Most fit=" + str(population[0].fitness) + " vs least fit=" + str(population[len(population) - 1].fitness))

    #write top founder_size pops to file
    for p in range(founder_size):
        netfile = output_dir + "/net/" + str(p) + ".txt"
        with open(netfile, 'wb') as net_out:
            nx.write_edgelist(population[p].net, net_out)   #make sure overwrites
        #also write info like population[p].fitness somewhere?

    output.outro_csv(output_dir, gens, output_freq)

    print("Worker " + str(worker_ID) + " finished.")



def evolve_master(population, subpop_gens, num_survive, crossover_percent, mutation_chance, output_dir,  pressure, tolerance, knapsack_solver):
    pop_size = len(population)

    output_dir += "/master/"
    #breed(population, num_survive, pop_size, crossover_percent, mutation_chance)

    for g in range(subpop_gens):
        max_RGGR = max_ETB = 0

        for p in range(pop_size):
            if (grow_freq != 0 and g % int(gens*grow_freq)):
                grow(population[p].net, 10, avg_degree)  #net, startsize, avgdegree
                #choice of GROWTH FUNCTION
                #g%int(math.log(grow_freq)) == 0): ?
                if (p == 0): console_report(population[0])

            population[p].fitness_parts = pressurize(population[p].net, pressure, tolerance, knapsack_solver)
            max_RGGR = max(max_RGGR, population[p].fitness_parts[0])
            max_ETB = max(max_ETB, population[p].fitness_parts[1])
            #print(population[p].fitness_parts)

        for p in range(pop_size):
            population[p].fitness = .5*population[p].fitness_parts[0]/max_RGGR + .5*population[p].fitness_parts[1]/max_ETB
            #might want to check that max_RGGR and max_ETB != 0 and throw warning if are (and divide by 1)

        breed(population, num_survive, pop_size, crossover_percent, mutation_chance)
        if (output_freq != 0 and g % int(gens/output_freq) == 0): output.to_csv(population, output_dir)  # need to write output.worker_csv()

    population.sort(key=operator.attrgetter('fitness'))
    #as in breed(), check that sorts with MAX fitness first

    return population



def grow(net, startSize, avg_degree):
    #operates only on nodes
    #adds edges to and from that node, instead of random
    #later change to avg_in_deg vs _out_deg

    # add numbered node
    node2 = node = len(net.nodes())
    net.add_node(node) #not rand assignment

    if (len(net.nodes()) > 3):
        for i in range(avg_degree):
            while (node2 == node):
                node2 = random.randint(0, len(net.nodes()) - 1)
            sign = random.randint(0, 1)
            if (sign == 0):     sign = -1

            direction = random.random()             #rand in or out
            if (direction < .5):
                net.add_edge(node2, node, sign=sign)
            else:
                net.add_edge(node, node2, sign=sign)


    #keep growing if < 2 nodes
    while (len(net.nodes()) < startSize): grow(net, startSize, avg_degree)


    #ASSUMPTION: net should remain connected
    connect_components(net)


def mutate(net):
    #operates on edges
        numEdges = len(net.edges())
        #mutation options: rm edge, add edge, change edge sender or target, change edge sign
        mut_type = random.random()
        if (mut_type < .25):
            # rm edge
            edges = net.edges()
            edge = edges[random.randint(0,numEdges-1)]
            net.remove_edge(edge[0], edge[1])  #might have to be *edge

            # ASSUMPTION: net should remain connected
            if (len(list(nx.weakly_connected_components(net))) > 1): add_edge(net)
            #print("after mutn type 1 : " + str(len(net.nodes())))

        elif(mut_type < .5):
            #add edge
            add_edge(net)
            #print("after mutn type 2 : " + str(len(net.nodes())))

        elif(mut_type < .75):
            #change an edge node
            edges = net.edges()
            edge = edges[random.randint(0,numEdges-1)]
            node = random.randint(0, len(net.nodes())-1)

            if (random.random() < .5):
                edge = [edge[0], node]
            else:
                edge = [node, edge[1]]

            while (len(list(nx.weakly_connected_components(net))) > 1): add_edge(net)
            #print("after mutn type 3 : " + str(len(net.nodes())))

        else:
            #change edge sign
            edges = net.edges()
            edge = edges[random.randint(0,numEdges-1)]
            net[edge[0]][edge[1]]['sign'] = -1*net[edge[0]][edge[1]]['sign']
            #print("after mutn type 4 : " + str(len(net.nodes())))


def pressurize(net, pressure, tolerance, knapsack_solver):
    #does all the reducing to kp and solving
    #calls calc_fitness()

    RGGR = ETB = QOS = ETR = 0
    # red-green/grey ratio, effective total benefits, quality of optimal solution, effective total ratio

    #dynam changes these to match net size
    #IF net sizes are all the same, move outside of pressurize() loop
    num_samples_relative = min(configs['sampling_rounds_max'], len(net.nodes()) * configs['sampling_rounds'])
    #curr min( config_max, #nodes+#edges)
    pressure_relative = int(pressure * len(net.nodes()))
    kp_instances = reducer.reverse_reduction(net, pressure_relative, int(tolerance), num_samples_relative, configs['advice_upon'], configs['biased'], configs['BD_criteria'])
    # ignore these parameters: configs['advice_upon'], configs['biased'], configs['BD_criteria']

    i = 1
    for kp in kp_instances:
        # in each iteration, a new knapsack instances is 'yeilded' by reverse_reduction function
        a_result = solver.solve_knapsack(kp, knapsack_solver)
        instance_RGGR = instance_ETB = instance_QOS = instance_ETR = 0

        i += 1
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

        if len(a_result) > 0:
            # print ("knapsack value = "+str(a_result[0]).ljust(15,' ')+"knapsack weight "+str(a_result[1]))
            Gs, Bs, Ds, Xs = [], [], [], []
            # -------------------------------------------------------------------------------------------------
            GENES_in, GENES_out, coresize, execution_time = a_result[4], a_result[5], a_result[9], a_result[10]
            total_benefit = a_result[0]
            total_dmg = a_result[1]
            num_green = len(a_result[6])
            num_red = len(a_result[7])
            num_grey = len(a_result[8])

            # -------------------------------------------------------------------------------------------------
            for g in GENES_in:  # notice that green_genes is a subset of GENES_in
                Gs.append(
                    str(g[0]) + '$' + str(net.in_degree(g[0])) + '$' + str(net.out_degree(g[0])))
                Bs.append(g[1])
                Ds.append(g[2])
                Xs.append(1)
            for g in GENES_out:  # notice that red_genes is a subset of GENES_out
                Gs.append(
                    str(g[0]) + '$' + str(net.in_degree(g[0])) + '$' + str(net.out_degree(g[0])))
                Bs.append(g[1])
                Ds.append(g[2])
                Xs.append(0)

            # Gs, Bs, Ds, Xs are, respectively,
            # the genes
            # their corresponding benefits
            # their corresponding weights (damages)
            # the solution vector (a binary 0/1 sequence, 0 = outside knapsack, 1=inside knapsack)

            soln_bens = []
            soln_ratios = []
            for g in range(len(Bs)):
                if (Xs[g] == 1):
                    if (Ds[g] != 0):
                        soln_ratios.append(Bs[g] / Ds[g])
                    else:
                        soln_ratios.append(Bs[g])
                    soln_bens.append(Bs[g])

            instance_ETB = sum(set(soln_bens))
            instance_ETR = sum(set(soln_ratios))
            instance_QOS = sum(soln_bens)

            if (num_grey != 0):
                instance_RGGR = (num_green + num_red) / num_grey
            else:
                instance_RGGR = (num_green + num_red)

        else:
            instance_fitness_ratio = 100
            instance_fitness_packedRatio = 100
            print ("WARNING in pressurize(): no results from oracle advice")

        ETB += instance_ETB
        ETR += instance_ETR
        RGGR += instance_RGGR
        QOS += instance_QOS


    ETB /= num_samples_relative
    ETR /= num_samples_relative
    RGGR /= num_samples_relative
    QOS /= num_samples_relative

    return [RGGR, ETB]


def console_report(net):
    #eventually as print to file?
    print ("\nNet size = " + str(len(net.net.nodes())))


def unused():
    #storage for unused functions
    for g in range(gens):
        #UNPARALLIZED


        '''PRE PARALLEL
        output.init_csv(pop_size, num_workers, output_dir, configs)
        evolve_master(population, merge_gens, num_survive, crossover_percent, mutation_chance, output_dir, pressure, tolerance, knapsack_solver)

        print ("\nEvolution starting...")
        for g in range(gens):

            for n in range(num_workers):
                population = [Net(M.copy(), i) for i in range(pop_size)]
                evolve_worker(n, population, num_workers, subpop_gens, num_survive,crossover_percent, mutation_chance,output_dir,  pressure, tolerance, configs)


            print("Master gen " + str(g) + " starting.")

            #read in nets from files
            for n in range(num_workers):
                in_dir = output_dir + str(n)
                for s in range(subpop_size):
                    netfile = in_dir + "/net/" + str(s) + ".txt"    #depends on output.worker_csv() format
                    population[n*subpop_size+s].net = nx.read_edgelist(netfile, create_using=nx.DiGraph())
                    #change population fitnesses, ect
                    #diff size nets due to unevolved nets, but should breed()

        #evolve_popn before giving back out to workers i think
        evolve_master(population, merge_gens, num_survive, crossover_percent, mutation_chance, output_dir,  pressure, tolerance, knapsack_solver)
        #(population, subpop_gens, num_survive, crossover_percent, mutation_chance, output_dir,  pressure, tolerance, knapsack_solver)
        '''

        ''' PARALLEL VERSION
        #handle not int cases, warning -> change params, or diff size pops
        pool = mp.Pool(num_workers)
        args = []
        barrier= mp.Barrier(num_workers-1) #why does -1 seem to work?

        for n in range(num_workers):
            #subpop = mp.Process(target=evolve_population_worker, args=(n, population[subpop_size*n:subpop_size*[n+1]], num_workers, subpop_gens, num_survive,crossover_percent, mutation_chance,output_dir,  pressure, tolerance, knapsack_solver))
                #check subpopn indices
                #shitload of params, maybe way to pass a bunch initially and only subset as time goes on
                #params: (worker_ID, founder_pop, num_workers, subpop_gens, num_survive, crossover_percent, mutation_chance,output_dir)
            args.append([n, population[subpop_size*n:subpop_size*(n+1)], num_workers, subpop_gens, num_survive,crossover_percent, mutation_chance,output_dir,  pressure, tolerance, configs])


        pool.starmap(evolve_worker, args)

        #WAIT for all jobs to finish (synch pt/bottleneck)
        #barrier.wait()
        pool.join()
        pool.close()
        '''


def parallel_param_test(configs):
    gens = int(configs['generations'])
    crossover_percent = float(configs['crossover_percent'])
    avg_degree = int(configs['average_degree'])
    output_file = configs['output_file']
    output_dir_orig = configs['output_directory'].replace("v4nu_minknap_1X_both_reverse/", '')  #no idea where this is coming from
    grow_freq = float(configs['growth_frequency'])
    output_freq = float(configs['output_frequency'])
    num_workers = int(configs['number_of_workers'])

    pressures = [50,100]
    tolerances = [10,50]
    pop_sizes = [20,100]
    mutation_freqs = [1,10]
    percent_survives = [10,20]
    start_sizes = [20,100]
    # total num params = 64

    for k in range(len(pop_sizes)):
        pop_size = pop_sizes[k]
        output_dir = output_dir_orig + "popSize" + str(pop_sizes[k]) + "/"

        output.init_csv(pop_sizes, num_workers, output_dir, configs)
        # configs produces are not so relevant here, although some params not tuned for will be shown

        pool = mp.Pool(num_workers)
        args = []

        for i in range(len(pressures)):
            for j in range(len(tolerances)):
                for l in range(len(mutation_freqs)):
                    for m in range(len(percent_survives)):
                        for n in range(len(start_sizes)):

                            ID = (i*16+j*8+l*4+m*2+n)
                            num_survive = int(pop_size*percent_survives[m]/100)
                            population = [Net(M.copy(), i) for i in range(pop_size)]

                            for p in range(pop_sizes[k]):
                                grow(population[p].net, start_sizes[n], avg_degree)
                            population_copy = [population[p].copy() for p in range(pop_size)]

                            output.parallel_configs(ID, output_dir, pressures[i], tolerances[j], pop_sizes[k], mutation_freqs[l], percent_survives[m], start_sizes[n])
                            worker_args = [ID, population_copy, num_workers, gens, num_survive,crossover_percent, mutation_freqs[l], output_dir,  pressures[i]/100, tolerances[j], configs]
                            args.append(worker_args)

        print("Starting parallel parameter search # " + str(k) + ".")
        pool.starmap(evolve_worker, args)

        print("Done evolving set # " + str(k) + ", generating images.")
        plot_nets.features_over_time(output_dir,num_workers,gens,pop_size, output_freq)

    print("Done.")




if __name__ == "__main__":

    config_file         = util.getCommandLineArgs ()
    M, configs          = init.initialize_master (config_file, 0)
    #print ("\nMain net begins with "+str(len(M.nodes())) +" nodes (genes) " + str(len(M.edges())) + " edges (interactions) ")

    #can call parallel_param_test here i think
    parallel_param_test(configs)

    '''
    #init config vals
    #a lot are modified from orig to allow dynamic resizing with diff net sizes
    knapsack_solver     = cdll.LoadLibrary(configs['KP_solver_binary'])
    pressure            = math.ceil ((float(configs['PT_pairs_dict'][1][0])/100.0))  #why not just pressure? SHOULD CHANGE
    tolerance           = configs['PT_pairs_dict'][1][1]
    gens = int(configs['generations'])
    pop_size = int(configs['population_size'])
    population = [Net(M.copy(),i) for i in range(pop_size)]
    fitness = [0 for i in range(pop_size)]  #unused i think
    survive_fraction = float(configs['percent_survive'])/100
    num_survive = int(pop_size*survive_fraction)

    num_samples = configs['sampling_rounds']
    max_samples = configs['sampling_rounds_max']
    mutation_freq = int(configs['mutation_percent'])
    crossover_percent = float(configs['crossover_percent'])
    avg_degree = int(configs['average_degree'])
    output_file = configs['output_file']
    output_dir = configs['output_directory'].replace("v4nu_minknap_1X_both_reverse/", '')  #no idea where this is coming from
    grow_freq = float(configs['growth_frequency'])
    output_freq = float(configs['output_frequency'])
    start_size = int(configs['starting_size'])

    num_workers = int(configs['number_of_workers'])
    #num_workers = mp.cpu_count() #override
    subpop_size = int(pop_size / num_workers)

    output.init_csv(pop_size, num_workers, output_dir, configs)
    #evolve_master(population, merge_gens, num_survive, crossover_percent, mutation_chance, output_dir, pressure, tolerance, knapsack_solver)

    print ("\nEvolution starting...")
    for p in range(pop_size):
        grow(population[p].net, start_size, avg_degree)

    for n in range(num_workers):
        population_copy = [population[p].copy() for p in range(pop_size)]
        evolve_worker(n, population_copy, num_workers, gens, num_survive,crossover_percent, mutation_freq,output_dir,  pressure, tolerance, configs)

    plot_nets.features_over_time(output_dir,num_workers,gens,pop_size, output_freq)
    #degree_dist.py
    #write output to final file?
    '''

    print ("Done.")



