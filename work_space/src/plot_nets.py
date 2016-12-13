#!/usr/bin/python3
import math, matplotlib, os
import matplotlib.pyplot as plt

matplotlib.use('Agg') # This must be done before importing matplotlib.pyplot
import numpy as np


#IMAGE GENERATION FNS()
def features_over_time(dirr, num_workers,gens, num_indivs, output_freq):
    #given dirr generates graphs
    worker_info, titles = parse_worker_info(dirr, num_workers,gens,num_indivs, output_freq)
    #later incld master
    init_img(dirr, num_workers)

    #master plots
    '''
    m_dirr = dirr + "/images/master/"
    for i in range(1, len(titles) - 1):
        for j in range(num_indivs):
            data = []
            for g in range(gens):
                data.append(master_info[g, j, i])  # titles are one off since net size not included
            plt.plot(data)
        plt.savefig(m_dirr + str(titles[i]))
        plt.clf()
    '''

    #worker plots
    for w in range(num_workers):
        w_dirr = dirr +"/" + str(w) + "/images/"
        for i in range(1,len(titles)):
            for j in range(num_indivs):
                data = []
                for g in range(int(gens*output_freq)):
                    data.append(worker_info[w,g,j,i-1]) #titles are one off since net size not included
                #print(data)
                plt.plot(data)
            plt.savefig(w_dirr + str(titles[i]))
            plt.clf()


    return


def fitness_over_params(dirr, num_workers):
    feature_info, titles = merge_paramtest(dirr, num_workers)
    num_features = len(titles)
    init_img(dirr, num_workers)
    for i in range (num_features):
        x = []
        y = []
        for w in range(num_workers):
            x.append(w)
            y.append(feature_info[w][i])
        plt.bar(x,y, align="center")
        plt.savefig(dirr + "/param_images/" + str(titles[i]) + ".png")
        plt.clf()
    x=[]
    y=[]
    for w in range(num_workers):  #spc plot, may be off if output.outro_csv() changes
        x.append(w)
        y.append(feature_info[w][1] + feature_info[w][3])
    plt.bar(x,y, align="center")
    plt.savefig(dirr + "/param_images/AvgChangeOfFitness_summedAfter.png")
    plt.clf()


#HELPER FNS()
def init_img(dirr, num_workers):
    if not os.path.exists(dirr + "/master/images"):
        os.makedirs(dirr + "/master/images")
    if not os.path.exists(dirr + "/param_images"):
        os.makedirs(dirr + "/param_images")
    for w in range(num_workers):
        if not os.path.exists(dirr +"/" + str(w) + "/images/" ):
            os.makedirs(dirr +"/" + str(w) + "/images/")


def merge_paramtest(dirr, num_workers):
    #curr naive output

    for w in range(num_workers):
        configs_file = dirr + "/" + str(w) + "/worker_configs.csv"
        outro_file = dirr + "/" + str(w) + "/outro_info.csv"

        with open(outro_file, 'r') as outro:
            if (w==0):
                titles = outro.readline().split(",")
                piece = titles[-1].split("\n")
                titles[-1] = piece[0]
                num_features = len(titles)
                feature_info = np.empty((num_workers, num_features))
            else: next(outro)
            features = outro.readline().split(",")
            piece = features[-1].split("\n")
            features[-1] = piece[0]
            #print("merge_paramtest() features: " + str(features))
            feature_info[w] = features
    return feature_info, titles



def parse_info(dirr, num_workers, gens, num_indivs):
    #returns 4D array with [worker#][gen][indiv#][feature]

    worker_info = None
    master_info = None
    titles = None
    num_features = 0 #assumes same num features in worker and master
    #master_info = list(csv.reader(open(dirr+"/master/info.csv"),'r'))

    with open(dirr+"/master/info.csv",'r') as info_csv:
        titles = info_csv.readline().split(",")
        num_features = len(titles)-1
        master_info = np.empty((gens, num_indivs, num_features))  # could print this info to 2nd line of file

        for i in range(num_indivs*gens):
            row = info_csv.readline().split(",", num_features)
            master_info[math.floor(i/num_indivs)][int(row[0])] = row[0:num_features]  #not sure why need to index whole array

    for w in range(num_workers):
        worker_info = np.empty((num_workers, gens, num_indivs, num_features))

        with open(dirr + "/" + str(w) + "/info.csv", 'r') as info_csv:
            next(info_csv)

            for i in range(num_indivs * gens):
                row = info_csv.readline().split(",", num_features)
                print(w,math.floor(i / num_indivs),int(row[0]))
                worker_info[w][math.floor(i / num_indivs)][int(row[0])] = row[0:num_features]  # not sure why need to index whole array



    return master_info, worker_info, titles



def parse_worker_info(dirr, num_workers, gens, num_indivs, output_freq):
    #returns 4D array with [worker#][gen][indiv#][feature]

    worker_info = None
    titles = None
    num_features = 0 #assumes same num features in worker and master
    #master_info = list(csv.reader(open(dirr+"/master/info.csv"),'r'))

    for w in range(num_workers):

        with open(dirr + "/" + str(w) + "/info.csv", 'r') as info_csv:
            if (w==0):
                titles = info_csv.readline().split(",")
                piece = titles[-1].split("\n")
                titles[-1] = piece[0]
                num_features = len(titles)-1
                worker_info = np.empty((num_workers, int(gens*output_freq), num_indivs, num_features))
            else:
                next(info_csv)

            for i in range(num_indivs * int(gens*output_freq)):
                row = info_csv.readline().split(",", num_features) #might be num_features -1 now
                piece = row[-1].split("\n")
                row[-1] = piece[0]
                worker_info[w][math.floor(i / num_indivs)][int(row[0])] = row[1:]  # not sure why need to index whole array
            #print(worker_info[1][:][0])


    return worker_info, titles


if __name__ == "__main__":
    dirr = "/home/2014/choppe1/Documents/EvoNet/work_space/data/output/partial_growth_run/v1.2/popSize20"

    features_over_time(dirr,16,500,20,1)
    #(dirr, num_workers,gens, num_indivs, output_freq)
    #fitness_over_params(dirr, 3)