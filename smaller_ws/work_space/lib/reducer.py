import random, util, math, time, numpy as np

#--------------------------------------------------------------------------------------------------  
def reverse_reduction(M, sample_size, T_percentage, advice_sampling_threshold, advice_upon, biased, BD_criteria):
    #print ("in reducer, " + str(advice_sampling_threshold))
    if  advice_sampling_threshold <=0:
        print ("WARNING: reverse_reduction yields empty set.")
        yield [{},{},0]
    else:      
        for i in range(advice_sampling_threshold):
            yield [
                    BDT_calculator_node_both   (M, util.advice_nodes (M, util.sample_p_elements(M.nodes(),sample_size), biased), T_percentage)
                  ]    
#--------------------------------------------------------------------------------------------------                
def BDT_calculator_node_both (M, Advice, T_percentage):
    BENEFITS, DAMAGES = {}, {}
    for target in Advice.keys():
        for source in M.predecessors (target):
            if M[source][target]['sign']==Advice[target]:      #in agreement with the Oracle
                ######### REWARDING the source node ###########
                if source in BENEFITS.keys():
                    BENEFITS[source]+=1
                else:
                    BENEFITS[source]=1
                    if source not in DAMAGES.keys():
                        DAMAGES[source]=0
                ######### REWARDING the target node ###########
                if target in BENEFITS.keys():
                    BENEFITS[target]+=1
                else:
                    BENEFITS[target]=1
                    if target not in DAMAGES.keys():
                        DAMAGES[target]=0
                ###############################################
                ###############################################
            else:                                              #in disagreement with the Oracle
                ######### PENALIZING the source node ##########
                if source in DAMAGES.keys():
                    DAMAGES[source]+=1
                else:
                    DAMAGES[source]=1
                    if source not in BENEFITS.keys():
                        BENEFITS[source]=0
                ######### PENALIZING the target node ##########
                if target in DAMAGES.keys():
                    DAMAGES[target]+=1
                else:
                    DAMAGES[target]=1
                    if target not in BENEFITS.keys():
                        BENEFITS[target]=0
                ###############################################

    T_edges = round (max (1, math.ceil (sum(DAMAGES.values())*(T_percentage/100))))

    assert len(BENEFITS.keys())==len(DAMAGES.keys())
    return BENEFITS, DAMAGES, T_edges
#--------------------------------------------------------------------------------------------------