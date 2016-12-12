#!/usr/local/bin/python3
import os, csv
import networkx as nx
import numpy as np


def parallel_configs(ID, output_dir, pressure, tolerance, pop_size, mutation_freq, percent_survive, start_size):

    title = "Pressure, Tolerance, Population Size, Mutation Frequency, Percent Survive, Starting Size\n"

    if not os.path.exists(output_dir + "/" + str(ID)):
        os.makedirs(output_dir)
    with open(output_dir + "/" + str(ID) + "/worker_configs.csv", 'w') as outfile:
        outfile.write(title)
        outfile.write(str(pressure) + "," + str(tolerance) + "," + str(pop_size) + "," + str(mutation_freq) + "," + str(percent_survive) + "," + str(start_size))

def net(ID, indiv, out_dir):
    if (indiv.net.edges()):
        output_file = out_dir + "/net_" + str(ID) + ".txt"
        with open(output_file , 'wb') as outfile:               #hopefully overwrites each time
            nx.write_edgelist(indiv.net, output_file)

        info_file = out_dir + "/fitness" + str(ID) + ".txt"
        with open(info_file , 'w') as outfile:
            outfile.write(indiv.fitness + "," + indiv.fitness_parts[0] + "," + indiv.fitness_parts[1])


def net_for_degree_dist(ID, population, out_dir):
    #file name: net#/rebirth# - gen

    for p in range(len(population)):
        if (population[0].net.edges()):
            output_file = out_dir + "/net" + str(ID) +  "_" + str(len(population[p].net.nodes())) + ".txt"
            with open(output_file , 'w') as outfile:
                outfile.write("From\tTo\tSign\n")
                for edge in population[p].net.edges():
                    sign = population[p].net[edge[0]][edge[1]]['sign']
                    if (sign == 1): sign='+'
                    elif (sign== -1):   sign = '-'
                    else: print("ERROR output(): unknown sign")
                    formatted = str(edge[0]) + "\t" + str(edge[1]) + "\t" + str(sign) + "\n"
                    outfile.write(formatted)
                #nx.write_edgelist(population[p].net, output_file)  #prob faster, but doesn't match degree_dist.py expectation

def init_csv(pop_size, num_workers, out_dir, configs):

    ''' for sep file for each net
    for p in range(pop_size):
        dir = out_dir + "net" + str(p) + "/"
        if not os.path.exists(dir):  #possible race condition in parallel
            os.makedirs(dir)
    '''
    #CHANGE header info to match to_csv

    csv_title = "Net ID#, Size, Fitness, Red-Green to Grey Ratio, Effective Total Benefits, Avg Degree\n"
    csv_popn_title = "Net Size, RGGR max, RGGR min, ETB max, ETB min\n"

    if not os.path.exists(out_dir+"/master/"):
        os.makedirs(out_dir+"/master/")
        with open(out_dir+"/master/info.csv",'w') as csv_out:
            csv_out.write(csv_title)

    for w in range(num_workers):

        worker_dir = out_dir + "/" + str(w) + "/"
        if not os.path.exists(worker_dir):
            os.makedirs(worker_dir)
        if not os.path.exists(worker_dir + "/net"):
            os.makedirs(worker_dir + "/net")
        with open(worker_dir+"info.csv", 'w') as csv_out:
            csv_out.write(csv_title)
        with open(worker_dir + "population_info.csv", 'w') as csv_out:
            csv_out.write(csv_popn_title)

    '''
    for p in range(pop_size):
        out_csv = out_dir + "/info_" + str(p) + ".csv"
        with open(out_csv, 'w') as outCsv:
            outCsv.write("Gen,Size,Parent1,Parent2,Rank,Fitness,Red-Green/Grey Ratio,Effective Total Benefits,Solution Sum Benefits/Net Size, Degree Distribution\n")
    '''

    out_configs = out_dir + "/configs_used.csv"

    with open(out_configs, 'w') as outConfigs:
        outConfigs.write("Pressure,Tolerance,Max Sampling Rounds, Generations, Population Size, Percent Survive, Mutation Percent, Crossover Percent, Starting Size\n")
        outConfigs.write(str(configs['pressure']) + "," + str(configs['tolerance']) + "," + str(configs['sampling_rounds_max']) + "," + str(configs['generations']) + "," + str(configs['population_size'])+ "," + str(configs['percent_survive']) + "," + str(configs['mutation_frequency']) + "," + str(configs['crossover_percent']) + "," + str(configs['starting_size']) + "\n")


def outro_csv(output_dir, gens, output_freq):

    with open(output_dir + "/population_info.csv", 'r') as popn_file:
        titles = popn_file.readline().split(",")
        piece = titles[-1].split("\n")
        titles[-1] = piece[0]
        num_features = len(titles)

        lines = []
        for i in range(int(gens*output_freq)):
            line = popn_file.readline().split(",")
            piece = line[-1].split("\n")
            line[-1] = piece[0]
            lines.append(line)

        avg_change = [0 for j in range(num_features)]

        for i in range(len(lines)-1):
            for j in range(num_features):
                avg_change[j] += (float(lines[i+1][j]) - float(lines[i][j])) / (gens*output_freq)
        for j in range(num_features):
            avg_change[j] /= len(lines)-1

    with open(output_dir + "/outro_info.csv", 'w') as outro_file:
        output = csv.writer(outro_file)
        row = []
        for j in range(num_features): #hopefully num_features in conserved from prev block
            row.append("Avg change of " + titles[j])
        output.writerow(row)
        row = []
        for j in range(num_features):
            row.append(str(avg_change[j]))
        output.writerow(row)


def to_csv(population, output_dir, popn_info):
    #might need to format into matrix instead of single col

    if (population[0].net.edges()):
        output_csv = output_dir + "/info.csv"

        with open(output_dir + "/population_info.csv", 'a') as popn_file:
            output = csv.writer(popn_file)
            row = []
            for i in range(len(popn_info)):
                row.append(popn_info[i])
            output.writerow(row)

        with open(output_csv, 'a') as output_file:
            output = csv.writer(output_file)


            if (len(population[0].net.nodes()) != len(population[1].net.nodes())):
                print("WARNING output(): nets are not same size. " +
                      str(len(population[0].net.nodes())) + ", " + str(len(population[1].net.nodes())))

            for p in range(len(population)):
                net_info = []
                net_info.append(population[p].id)
                net_info.append(len(population[p].net.nodes()))
                net_info.append(population[p].fitness)
                net_info.append(population[p].fitness_parts[0])
                net_info.append(population[p].fitness_parts[1])
                net_info.append(sum(population[p].net.degree().values())/len(population[p].net.nodes()))
                output.writerow(net_info)
                #more concisely?


                #deg distrib
                ''' There is a much shorter way
                num_out_edges = []
                out_edge_freq = []
                num_in_edges = []
                in_edge_freq = []
                for node in population[p].net.nodes():
                    num_out = len(population[p].net.out_edges(node))
                    num_in = len(population[p].net.in_edges(node))
                    if (num_out not in num_out_edges):
                        num_out_edges.append(num_out)
                        out_edge_freq.append(1)
                    else:
                        out_edge_freq[num_out_edges.index(num_out)] += 1

                    if (num_in not in num_in_edges):
                        num_in_edges.append(num_in)
                        in_edge_freq.append(1)
                    else:
                        in_edge_freq[num_in_edges.index(num_in)] += 1
                out_edge_freq.sort()
                in_edge_freq.sort()
                net_info.append(out_edge_freq)
                net_info.append(str(in_edge_freq) + "\n")
                #net_info.append("\n")
                #check that these are in single cells
                '''
            #might need /n

