#!/usr/local/bin/python3
import math, matplotlib, os
import matplotlib.pyplot as plt

matplotlib.use('Agg') # This must be done before importing matplotlib.pyplot
import numpy as np


def features_over_time(dir, num_workers,gens, num_indivs, output_freq):
    #given dir generates graphs
    worker_info, titles = parse_worker_info(dir, num_workers,gens,num_indivs, output_freq)
    #later incld master
    init_img(dir, num_workers)

    #master plots
    '''
    m_dir = dir + "/images/master/"
    for i in range(1, len(titles) - 1):
        for j in range(num_indivs):
            data = []
            for g in range(gens):
                data.append(master_info[g, j, i])  # titles are one off since net size not included
            plt.plot(data)
        plt.savefig(m_dir + str(titles[i]))
        plt.clf()
    '''

    #worker plots
    for w in range(num_workers):
        w_dir = dir + "/images/" + str(w) + "/"
        for i in range(1,len(titles)):
            for j in range(num_indivs):
                data = []
                for g in range(int(gens*output_freq)):
                    data.append(worker_info[w,g,j,i-1]) #titles are one off since net size not included
                #print(data)
                plt.plot(data)
            plt.savefig(w_dir + str(titles[i]))
            plt.clf()


    return


def init_img(dir, num_workers):
    if not os.path.exists(dir + "/images/master"):
        os.makedirs(dir + "/images/master")
    for w in range(num_workers):
        if not os.path.exists(dir + "/images/" + str(w)):
            os.makedirs(dir + "/images/" + str(w))

def parse_info(dir, num_workers, gens, num_indivs):
    #returns 4D array with [worker#][gen][indiv#][feature]

    worker_info = None
    master_info = None
    titles = None
    num_features = 0 #assumes same num features in worker and master
    #master_info = list(csv.reader(open(dir+"/master/info.csv"),'r'))

    with open(dir+"/master/info.csv",'r') as info_csv:
        titles = info_csv.readline().split(",")
        num_features = len(titles)-1
        master_info = np.empty((gens, num_indivs, num_features))  # could print this info to 2nd line of file

        for i in range(num_indivs*gens):
            row = info_csv.readline().split(",", num_features)
            master_info[math.floor(i/num_indivs)][int(row[0])] = row[0:num_features]  #not sure why need to index whole array

    for w in range(num_workers):
        worker_info = np.empty((num_workers, gens, num_indivs, num_features))

        with open(dir + "/" + str(w) + "/info.csv", 'r') as info_csv:
            next(info_csv)

            for i in range(num_indivs * gens):
                row = info_csv.readline().split(",", num_features)
                print(w,math.floor(i / num_indivs),int(row[0]))
                worker_info[w][math.floor(i / num_indivs)][int(row[0])] = row[0:num_features]  # not sure why need to index whole array



    return master_info, worker_info, titles



def parse_worker_info(dir, num_workers, gens, num_indivs, output_freq):
    #returns 4D array with [worker#][gen][indiv#][feature]

    worker_info = None
    titles = None
    num_features = 0 #assumes same num features in worker and master
    #master_info = list(csv.reader(open(dir+"/master/info.csv"),'r'))

    for w in range(num_workers):

        with open(dir + "/" + str(w) + "/info.csv", 'r') as info_csv:
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
    dir = "/Users/Crbn/Desktop/McG Fall '16/EvoNets/evoNet/work_space/data/output/v1.1"

    features_over_time(dir,2,1000,40,100)
    #(dir, num_workers,gens, num_indivs, output_freq)