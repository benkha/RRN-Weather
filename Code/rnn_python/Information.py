#Information Analysis
import entropy_estimators as ee
import numpy as np
import pickle
import threading
import networkx as nx
import matplotlib.pyplot as plt

def loadfile(name):
    pkl_file = open(name, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data

class MIThreat(threading.Thread):
    def __init__(self, x, y):
        threading.Thread.__init__(self)
        self.result = "None"
        self.x = x
        self.y = y

    def run(self):
        self.result = mutualInformation(self.x, self.y, 10)

def mutualInformation(x, y, k):
    return ee.mi(np.array([x]).T, np.array([y]).T, k=k)

def makeDataEqual(data):
    #This just breaks down the seperation of inputs, outputs, and the states of hidden neurons.
    data2 = []
    a = 0
    while a < len(data):
        data2.append([])
        b = 0
        while b < len(data[a][0]):
            data2[a].append(list(data[a][0][b]) + list(data[a][2][b]) + list(data[a][1][b]))
            b += 1
        a += 1
    return data2

def giveClippedAndDelayed(data):
    numberOfNodes = len(data[0][0])
    data2 = []
    a = 0
    while a < numberOfNodes:
        fullClipped = []
        fullDelayed = []
        b = 0
        while b < len(data):
            c = 0
            while c < (len(data[b]) - 1):
                fullClipped.append(data[b][c][a])
                fullDelayed.append(data[b][c+1][a])
                c += 1
            b += 1
        data2.append([fullClipped, fullDelayed])
        a += 1
    return data2

def calculateInformationTransfer(data):
    infoTransfer = []
    numberOfNodes = len(data)
    examplesToRunOn = 200 #should be len(data[0][0]) for the actual final results
    threads = []
    a = 0
    while a < (numberOfNodes-1):
        threads.append([])
        infoTransfer.append([])
        b = a + 1
        while b < numberOfNodes:
            thread1 = MIThreat(data[a][0][:examplesToRunOn], data[b][1][:examplesToRunOn])
            thread2 = MIThreat(data[a][1][:examplesToRunOn], data[b][0][:examplesToRunOn])
            thread1.start()
            thread2.start()
            threads[a].append([thread1, thread2])
            #aToB = mutualInformation(data[a][0], data[b][1], 10)
            #bToA = mutualInformation(data[a][1], data[b][0], 10)
            #infoTransfer[a].append([aToB, bToA])
            b += 1
        a += 1
    a = 0
    while a < len(threads):
        b = 0
        while b < len(threads[a]):
            t1 = threads[a][b][0]
            t2 = threads[a][b][1]
            t1.join()
            t2.join()
            infoTransfer[a].append([t1.result, t2.result])
            b += 1
        a += 1
    return infoTransfer

def makeInfoTransferList(infoTransfer):
    list1 = []
    a = 0
    while a < len(infoTransfer):
        b = 0
        while b < len(infoTransfer[a]):
            c = 0
            while c < len(infoTransfer[a][b]):
                list1.append(infoTransfer[a][b][c])
                c += 1
            b += 1
        a += 1
    return list1

def transfersAbove(infoTransfer, threshold1, threshold2):
    #max(threshold1, threshold2*infoTransfer[a][b][1])
    list1 = []
    a = 0
    while a < len(infoTransfer):
        b = 0
        while b < len(infoTransfer[a]):
            realb = a + b + 1
            if infoTransfer[a][b][0] > threshold1:
                list1.append([a, realb])
            if infoTransfer[a][b][1] > threshold1:
                list1.append([realb, a])
            b += 1
        a += 1
    return list1

def plotList(list1):
    list2 = []
    list3 = []
    for t in list1:
        list3.append(t[0])
        list3.append(t[1])
        list2.append((t[0], t[1]))

    G = nx.DiGraph()
    G.add_edges_from(list2)
    # Need to create a layout when doing
    # separate calls to draw nodes and edges
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size = 500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, arrows=True)
    plt.show()

def rankingAlgorithm(list1):
    def rankIteration(ranking, list1):
        newranking = []
        a = 0
        while a < len(ranking):
            num = 0.0
            total = 0.0
            b = 0
            while b < len(list1):
                key1 = list1[b][0]
                key2 = list1[b][1]
                if key1 == a:
                    total += (ranking[key2] - 1.0)
                    num += 1.0
                elif key2 == a:
                    total += (ranking[key1] + 1.0)
                    num += 1.0
                b += 1
            if num == 0.0:
                value = "None"
            else:
                value = total / num
            newranking.append(value)
            a += 1
        return newranking

    ranking = np.zeros(45)
    a = 0
    while a < 20:
        ranking = rankIteration(ranking, list1)
        a += 1
    return ranking

'''
data1 = loadfile('data/CS294Data/testValues1.pkl')
data2 = makeDataEqual(data1)
data2 = giveClippedAndDelayed(data2)
infoTransfer = calculateInformationTransfer(data2)
output = open('data/CS294Data/transfer_k200_1.pkl', 'wb')
pickle.dump(infoTransfer, output)
output.close()
'''
infoTransfer = loadfile('data/CS294Data/transfer_k200_1.pkl')
list1 = transfersAbove(infoTransfer, 0.2, 0.0)
ranking = rankingAlgorithm(list1)
print (ranking)
quit()



degs = np.zeros(25)
nums = []
a = 0
while a < len(list1):
    num1 = list1[a][0]-21
    num2 = list1[a][1]-21
    nums.append(num1)
    nums.append(num2)
    degs[num1] += 1
    degs[num2] += 1
    a += 1
print (degs)
print (np.unique(nums))
#plotList(list1)







quit()
list1 = makeInfoTransferList(infoTransfer)
list1 = sorted(list1)
#print (list1[(len(list1)-50):])
#print (infoTransfer[0])


quit()

print (np.array(data2).shape)
quit()
y0 = np.array(data1[0][2]).T
(y1, y2) = (y0[0], y0[1])
print (y1)
print (y2)

y1 = np.array(np.arange(0, 20, .1))
y2 = np.array(np.arange(0, 20, .1))
y1 = np.array([y1]).T
y2 = np.array([y2]).T
print (ee.entropy(y1, k=10))
print (ee.entropy(y2, k=10))
print (ee.mi(y1,y2, k=10))
quit()
#print (sklearn.feature_selection.mutual_info_regression(y2.reshape(-1, 1), y1.reshape(-1, 1)))
#print (sklearn.metrics.normalized_mutual_info_score(y1, y2))
#'''
#print (sklearn.metrics.mutual_info_score(y1, y2))
#print (sklearn.metrics.normalized_mutual_info_score(y1, y2))
mi1 = mutual_information_2d(y1, y2)

print (entropy(y1.reshape(-1, 1)))
print (entropy(y2.reshape(-1, 1)))
print (mi1)
#'''
