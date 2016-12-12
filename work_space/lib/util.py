import sys, random
#--------------------------------------------------------------------------------------------------
def getCommandLineArgs():
    if len(sys.argv) < 2:
        print ("Usage: python3 test.py [/absolute/path/to/configs/file.txt]\nExiting..\n")
        sys.exit()
    return [str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3])]  #used args 1,2, 3 instead of 0,1 due to pyCharm env
#----------------------------------------------------------------------------  
def slash(path):
    return path+(path[-1] != '/')*'/'
#--------------------------------------------------------------------------------------------------
def flip():
    return random.SystemRandom().choice([1,-1])
#--------------------------------------------------------------------------------------------------
def sample_p_elements (elements,p):
    #elements = nodes or edges
    return  random.SystemRandom().sample(elements,p) 
#--------------------------------------------------------------------------------------------------
def advice_nodes (M, sample_nodes, biased):
    advice = {}
    if not biased:
        for node in sample_nodes: 
            advice[node]=flip()
    else:
        for node in sample_nodes:
            biased_center       = 0.5 + M.node[node]['conservation_score']
            rand                = random.SystemRandom().uniform(0,1)
            if rand <= biased_center:
                advice[node] = 1    #should be promoted (regulation) or conserved (evolution)
            else:
                advice[node] = -1   #should be inhibited (regulation) or deleted (evolution)
    
    return advice
#--------------------------------------------------------------------------------------------------
