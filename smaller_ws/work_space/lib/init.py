import sys, os, csv, time, math, networkx as nx, shutil
sys.path.insert(0, os.getenv('lib'))
import util
#--------------------------------------------------------------------------------------------------
def initialize_master (cl_args, num_workers):   
    
    configs                        = load_simulation_configs (cl_args[2], 0)     #changed cl_args[1] -> [2]
    
    M                              = load_network    (configs) 
    configs['number_of_genes']     = len(M.nodes())   #WARNING: does not dynam adapt to net size, but curr unused
    configs['sampling_rounds']     = configs['sampling_rounds']
        #ORIG: min (configs['sampling_rounds_max'], (len(M.nodes())+len(M.edges()))*configs['sampling_rounds'])
            #now dynam adjusted to net size
    configs ['worker_load']        = int (math.ceil (float(configs['sampling_rounds']) / max(1, float(num_workers))))
    configs ['num_workers']        = num_workers
      
    return M, configs
#--------------------------------------------------------------------------------------------------    
def initialize_worker(cl_args):
    configs = load_simulation_configs (cl_args[1], -1)
    return configs
#--------------------------------------------------------------------------------------------------
def load_simulation_configs (param_file, rank):

    parameters = (open(param_file,'r')).readlines()
    assert len(parameters)>0
    configs = {}
    for param in parameters:
        if len(param) > 0: #ignore empty lines
            if param[0] != '#': #ignore lines beginning with #
                param = param.split('=')
                if len (param) == 2:
                    key   = param[0].strip().replace (' ', '_')
                    value = param[1].strip()
                    configs[key] = value
    
    #assert os.path.isfile(configs['network_file']) # the only mandatory parameters 
    configs['biased']              = configs['biased'] == 'True'
    bORu = 'u'
    if configs['biased']:
        bORu = 'b'
    configs['KP_solver_name']      = configs['KP_solver_binary'].split('/')[-1].split('.')[0]  
    configs['stamp']               = configs['version']+configs['advice_upon'][0]+bORu+'_'+ configs['KP_solver_name']+'_'+configs['sampling_rounds']+'_'+ configs['BD_criteria']+'_'+configs['reduction_mode'] 
    configs['timestamp']           = time.strftime("%B-%d-%Y-h%Hm%Ms%S")
    configs['pressure']            = [float(p) for p in configs['pressure'].split(',') ]        
    configs['tolerance']           = [float(t) for t in configs['tolerance'].split(',') ]    
    configs['sampling_rounds_nX']  = configs['sampling_rounds']
    configs['sampling_rounds']     = int(''.join([d for d in configs['sampling_rounds'] if d.isdigit()]))
    configs['sampling_rounds_max'] = int (configs['sampling_rounds_max'])      
    configs['output_directory']    = util.slash (util.slash (configs['output_directory']) +configs['stamp'])
    configs['stats_dir']           = configs['output_directory']+"00_network_stats/" 
    configs['datapoints_dir']      = configs['output_directory']+"02_raw_instances_simulation/data_points/"
    configs['params_save_dir']     = configs['output_directory']+"02_raw_instances_simulation/parameters/"
    configs['progress_file']       = configs['output_directory']+"02_raw_instances_simulation/progress.dat"
    configs['progress_dir']        = configs['output_directory']+"02_raw_instances_simulation/"
    configs['DUMP_DIR']            = util.slash (configs['output_directory'])+"dump_raw_instances"
    configs['alpha']               = float(configs['alpha']) 
    #--------------------------------------------
    index = 1
    ALL_PT_pairs = {}
    for p in sorted (configs['pressure']):
        for t in sorted (configs['tolerance']):
            ALL_PT_pairs[index] = (p,t)
            index+=1
    completed_pairs                = []
    if os.path.isdir (configs['datapoints_dir']):
        for r,ds,fs in os.walk(configs['datapoints_dir']):
            RAW_FILES       = [f for f in fs if 'RAW' in f]            
            for raw_file in RAW_FILES:
                #file names must be as such: Vinayagam_RAW_INSTANCES_p020.0_t001.0_V3_MINKNAP_4X_BOTH_SCRAMBLE_June-13-2016-h09m15s55.csv
                split = raw_file.split('_')
                p     = float(''.join([d for d in split[-8] if d.isdigit() or d=='.']))
                t     = float(''.join([d for d in split[-7] if d.isdigit() or d=='.']))
                completed_pairs.append((p,t))
    configs['PT_pairs_dict'] = {}
    for index in sorted(ALL_PT_pairs.keys()):
        if not ALL_PT_pairs[index] in completed_pairs:
            configs['PT_pairs_dict'][index] = ALL_PT_pairs[index]        
    #--------------------------------------------   
    if rank == 0: #only master should create dir, prevents workers from fighting over creating the same dir
        while not os.path.isdir (configs['output_directory']):
            print (configs['output_directory'])
            try:
                os.makedirs (configs['output_directory']) # will raise an error if invalid path, which is good
            except:
                time.sleep(5)
                print ("In load_simulation_configs(), rank=0, and Im still trying to create "+configs['output_directory']+" .. is this a correct path?")
                continue

    return configs
#--------------------------------------------------------------------------------------------------  
def load_network (configs):    
    edges_file = open (configs['network_file'],'r') #note: with nx.Graph (undirected), there are 2951  edges, with nx.DiGraph (directed), there are 3272 edges
    M=nx.DiGraph()     
    next(edges_file) #ignore the first line
    for e in edges_file: 
        interaction = e.split()
        assert len(interaction)>=2
        source, target = str(interaction[0]), str(interaction[1])
        if (len(interaction) >2):
            if (str(interaction[2]) == '+'):
                Ijk=1
            elif  (str(interaction[2]) == '-'):
                Ijk=-1
            else:
                print ("Error: bad interaction sign in file "+network_edge_file+"\nExiting...")
                sys.exit()
        else:
            Ijk=util.flip()     
        M.add_edge(source, target, sign=Ijk)    
    
    # conservation scores:
    if not configs['biased']:
        return M
    else:
        return conservation_scores (M, configs)
#--------------------------------------------------------------------------------------------------
def conservation_scores (M, configs):
    degrees           = [d for d in M.degree().values()]
    set_degrees       = list(set(degrees))
    frequencies       = {d:degrees.count(d) for d in set_degrees}
    mind              = min(set_degrees)
    N                 = M.number_of_nodes() 
    meand             = float(sum(degrees))/float(N)
    a, b              = 0, 0.5
    alpha             = configs['alpha']
    if configs['advice_upon'] == 'edges':
        for e in M.edges():
            source_degree                    = M.degree (e[0])            
            #M[e[0]][e[1]]['conservation_score']  = 1.0/float(frequencies[source_degree])
            M[e[0]][e[1]]['conservation_score']  = scale(source_degree,  N, meand,  a, b, alpha)
    elif configs['advice_upon'] == 'nodes':
        for n in M.nodes():
            degree = M.degree(n)
            if degree > 0:
                #M.node[n]['conservation_score']  = 1.0/float(frequencies[degree]) 
                M.node[n]['conservation_score']  = scale(degree, N, meand,  a, b, alpha)
            else:
                M.node[n]['conservation_score']  = 0 #island node
    else:
        print ("FATAL: unrecognized value for configs['advice_upon'] parameter\nExiting ..")
        sys.exit(1)
    return M
#--------------------------------------------------------------------------------------------------
def scale (d, N,  meand, a, b, alpha):
    if d <= meand:
        return 0
    numerator   = (b-a)*math.pow((d-meand),2)
    denumenator = N*b
    return  math.pow((float(numerator)/float(denumenator)) +a, alpha)
#--------------------------------------------------------------------------------------------------
