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

def savefile(name, data):
    output = open(name, 'wb')
    pickle.dump(data, output)
    output.close()

class MIThreat(threading.Thread):
    def __init__(self, x, y):
        threading.Thread.__init__(self)
        self.result = "None"
        self.x = x
        self.y = y

    def run(self):
        self.result = mutualInformation(self.x, self.y, 10)

class EntropyThread(threading.Thread):
    def __init__(self, x):
        threading.Thread.__init__(self)
        self.result = "None"
        self.x = x

    def run(self):
        self.result = calculateEntropy(self.x, 10)

class VectorMIThread(threading.Thread):
    def __init__(self, x, y):
        threading.Thread.__init__(self)
        self.result = "None"
        self.x = x
        self.y = y

    def run(self):
        self.result = ee.mi(self.x, self.y, k=10)

def mutualInformation(x, y, k):
    return ee.mi(np.array([x]).T, np.array([y]).T, k=k)

def calculateEntropy(x, k):
    return ee.entropy(np.array([x]).T, k=k)

def makeDataEqual(data):
    #This just breaks down the seperation of inputs, outputs, and the states of hidden neurons.
    data2 = []
    a = 0
    while a < len(data):
        data2.append([])
        b = 0
        while b < (len(data[a][0]) - 2):
            data2[a].append(list(data[a][0][b+2]) + list(data[a][2][b]) + list(data[a][1][b+1]))
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
    examplesToRunOn = 1000 #should be len(data[0][0]) for the actual final results
    threads = []
    a = 0
    while a < (numberOfNodes-1):
        threads.append([])
        infoTransfer.append([])
        b = a + 1
        while b < numberOfNodes:
            thread1 = MIThreat(data[a][0][:examplesToRunOn], data[b][1][:examplesToRunOn])
            thread2 = MIThreat(data[a][1][:examplesToRunOn], data[b][0][:examplesToRunOn])
            thread3 = MIThreat(data[a][0][:examplesToRunOn], data[b][0][:examplesToRunOn])
            thread1.start()
            thread2.start()
            thread3.start()
            threads[a].append([thread1, thread2, thread3])
            #aToB = mutualInformation(data[a][0], data[b][1], 10)
            #bToA = mutualInformation(data[a][1], data[b][0], 10)
            #infoTransfer[a].append([aToB, bToA])
            b += 1
        a += 1

    threads2 = []
    a = 0
    while a < numberOfNodes:
        thread4 = EntropyThread(data[a][0][:examplesToRunOn])
        thread4.start()
        threads2.append(thread4)
        a += 1

    a = 0
    while a < len(threads):
        b = 0
        while b < len(threads[a]):
            t1 = threads[a][b][0]
            t2 = threads[a][b][1]
            t3 = threads[a][b][2]
            t1.join()
            t2.join()
            t3.join()
            infoTransfer[a].append([t1.result, t2.result, t3.result])
            b += 1
        a += 1

    entropyList = []
    a = 0
    while a < len(threads2):
        t4 = threads2[a]
        t4.join()
        entropyList.append(t4.result)
        a += 1

    return (infoTransfer, entropyList)

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
    print ("nodes:" + str(len(np.unique(list3))))
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

def normalizedInfoTransfer(infoTransfer, entropyList):
    a = 0
    while a < len(infoTransfer):
        b = 0
        while b < len(infoTransfer[a]):
            realb = 1 + b + a
            minEntropy = min(entropyList[a], entropyList[realb])
            infoTransfer[a][b][0] = infoTransfer[a][b][0]/minEntropy
            infoTransfer[a][b][1] = infoTransfer[a][b][1]/minEntropy
            b += 1
        a += 1
    return infoTransfer

def nodesInTransferList(list1):
    nodes = []
    a = 0
    while a < len(list1):
        nodes.append(list1[a][0])
        nodes.append(list1[a][1])
        a += 1
    nodes = np.unique(nodes)
    return nodes

def entropyLowerBound(entropyList):
    min1 = 0.5
    '''
    entropyList = np.array(entropyList)
    min1 = np.min(entropyList)
    entropyList = entropyList + (min1 * -1.0) + 1.0
    return list(entropyList)
    '''
    a = 0
    while a < len(entropyList):
        entropyList[a] = max(entropyList[a], min1)
        a += 1
    return entropyList

def averageWithNone(list1):
    total = 0.0
    num = 0.0
    a = 0
    while a < len(list1):
        if list1[a] != "None":
            total += list1[a]
            num += 1.0
        a += 1
    avg = total / num
    return avg

def removeNone(list1):
    avg = averageWithNone(list1)
    a = 0
    while a < len(list1):
        if list1[a] == "None":
            list1[a] = avg
        a += 1
    return list1

def calculateAverageTransfer(infoTransfer, mode="total"):
    total = 0.0
    num = 0.0
    a = 0
    while a < len(infoTransfer):
        b = 0
        while b < len(infoTransfer[a]):
            if mode == "total":
                total += infoTransfer[a][b][0] + infoTransfer[a][b][1]
            elif mode == "net":
                total += abs(infoTransfer[a][b][0] - infoTransfer[a][b][1])
            num += 2.0
            b += 1
        a += 1
    avg = total / num
    return avg

def complexRankingAlgorithm(infoTransfer):
    def reformatInfoTransfer(infoTransfer, avg):
        infoTransfer2 = []
        a = 0
        while a < len(infoTransfer):
            b = 0
            while b < len(infoTransfer[a]):
                infoTransfer2.append([a, a + b + 1, infoTransfer[a][b][0] / avg, infoTransfer[a][b][1] / avg])
                b += 1
            a += 1
        return infoTransfer2

    def rankIteration(ranking, infoTransfer):
        newranking = []
        a = 0
        while a < len(ranking):
            num = 0.0
            total = 0.0
            b = 0
            while b < len(infoTransfer):
                key1 = infoTransfer[b][0]
                key2 = infoTransfer[b][1]
                if (key1 == a) or (key2 == a):
                    if (key2 == a):
                        key2 = key1
                        key1 = a
                    total += (ranking[key2] + infoTransfer[b][2] - infoTransfer[b][3])
                num += 1.0
                b += 1
            if num == 0.0:
                value = "None"
            else:
                value = total / num
            newranking.append(value)
            a += 1
        return newranking

    avg = calculateAverageTransfer(infoTransfer, mode="net")
    infoTransfer2 = reformatInfoTransfer(infoTransfer, avg)
    ranking = np.zeros(45)
    a = 0
    while a < 1:
        ranking = rankIteration(ranking, infoTransfer2)
        a += 1
    return ranking

def applySingleTimeMutualInformation(infoTransfer, mode="subtract"):
    a = 0
    while a < len(infoTransfer):
        b = 0
        while b < len(infoTransfer[a]):
            if mode == "subtract":
                infoTransfer[a][b][0] = infoTransfer[a][b][0] - infoTransfer[a][b][2]
                infoTransfer[a][b][1] = infoTransfer[a][b][1] - infoTransfer[a][b][2]
            del infoTransfer[a][b][-1]
            b += 1
        a += 1
    return infoTransfer

def mutualWithXY(data, noX=False):
    #If noX = True, the mutual information of the X neurons is not recorded
    #This takes unprocessed data
    def processData(data, noX=False):
        xData = []
        allData = []
        a = 0
        while a < len(data):
            b = 0
            while b < len(data[a][0]):
                xData.append(data[a][0][b])
                if noX:
                    allData.append(list(data[a][2][b]) + list(data[a][1][b]))
                else:
                    allData.append(list(data[a][0][b]) + list(data[a][2][b]) + list(data[a][1][b]))
                b += 1
            a += 1

        yData = loadfile('data/CS294Data/yData_testing1.pkl')
        allData = list(np.array(allData).T)
        return (xData, yData, allData)

    (xData, yData, allData) = processData(data, noX=noX)
    threads = []
    examplesToRunOn = 200
    a = 0
    while a < len(allData):
        thread1 = VectorMIThread(list(np.array([allData[a][:examplesToRunOn]]).T), xData[:examplesToRunOn])
        thread2 = VectorMIThread(list(np.array([allData[a][:examplesToRunOn]]).T), yData[:examplesToRunOn])
        thread1.start()
        thread2.start()
        threads.append([thread1, thread2])
        a += 1
    xMI = []
    yMI = []
    a = 0
    while a < len(threads):
        t1 = threads[a][0]
        t2 = threads[a][1]
        t1.join()
        t2.join()
        xMI.append(t1.result)
        yMI.append(t2.result)
        a += 1
    return (xMI, yMI)

def layerMutualWithXY(data):
    #This takes unprocessed data
    def processData(data):
        xData = []
        hData = []
        oData = []
        a = 0
        while a < len(data):
            b = 0
            while b < len(data[a][0]):
                xData.append(data[a][0][b])
                hData.append(data[a][1][b])
                oData.append(data[a][2][b])
                b += 1
            a += 1

        yData = loadfile('data/CS294Data/yData_testing1.pkl')
        #Input, Hidden, Output, True Y Values.
        return (xData, hData, oData, yData)

    (xData, hData, oData, yData) = processData(data)
    examplesToRunOn = 200
    Thread_hxMI = VectorMIThread(hData[:examplesToRunOn], xData[:examplesToRunOn])
    Thread_hyMI = VectorMIThread(hData[:examplesToRunOn], yData[:examplesToRunOn])
    Thread_oxMI = VectorMIThread(oData[:examplesToRunOn], xData[:examplesToRunOn])
    Thread_oyMI = VectorMIThread(oData[:examplesToRunOn], yData[:examplesToRunOn])
    Thread_hxMI.start()
    Thread_hyMI.start()
    Thread_oxMI.start()
    Thread_oyMI.start()
    Thread_hxMI.join()
    Thread_hyMI.join()
    Thread_oxMI.join()
    Thread_oyMI.join()
    hxMI = Thread_hxMI.result
    hyMI = Thread_hyMI.result
    oxMI = Thread_oxMI.result
    oyMI = Thread_oyMI.result
    return (hxMI, hyMI, oxMI, oyMI)

def saveRankings():
    infoTransfer = loadfile('data/CS294Data/transfer_k1000_6.pkl')
    infoTransfer = applySingleTimeMutualInformation(infoTransfer)
    list1 = transfersAbove(infoTransfer, 0.05, 0.0)
    cranking = complexRankingAlgorithm(infoTransfer)
    savefile('data/CS294Data/ranking_complex_1.pkl', cranking)
    sranking = rankingAlgorithm(list1)
    savefile('data/CS294Data/ranking_simple_1.pkl', sranking)
    nrsranking = removeNone(sranking)
    savefile('data/CS294Data/ranking_simple_removeNone_1.pkl', nrsranking)

def saveTransferAndEntropy():
    data1 = loadfile('data/CS294Data/testValues1.pkl')
    data2 = makeDataEqual(data1)
    data2 = giveClippedAndDelayed(data2)
    data2 = np.array(data2)
    (infoTransfer, entropyList) = calculateInformationTransfer(data2)
    output = open('data/CS294Data/transfer_k1000_6.pkl', 'wb')
    pickle.dump(infoTransfer, output)
    output.close()
    output = open('data/CS294Data/entropy_k1000_4.pkl', 'wb')
    pickle.dump(entropyList, output)
    output.close()

def saveTestingYData():
    data = loadfile('data/CS294Data/testValues1.pkl')
    yData = []
    a = 0
    while a < len(data):
        b = 0
        while b < len(data[a][0]):
            yData.append(data[a][2][b])
            b += 1
        a += 1
    savefile('data/CS294Data/yData_testing1.pkl', yData)

#################################################
###                                           ###
###       Functions for Final Processes       ###
###                                           ###
#################################################

def plotXYMutualInfo():
    data1 = loadfile('data/CS294Data/testValues1.pkl')
    (xMI, yMI) = mutualWithXY(data1)
    xMI = np.array(xMI)
    yMI = np.array(yMI)

    x_scatter = plt.scatter(xMI[:18], yMI[:18], label='Input Neurons')
    h_scatter = plt.scatter(xMI[20:], yMI[20:], label='Hidden Neurons')
    y_scatter = plt.scatter(xMI[18:20], yMI[18:20], label='Output Neurons')
    plt.legend(handles=[x_scatter, h_scatter, y_scatter])
    plt.show()

    x_scatter = plt.scatter(xMI[:18], yMI[:18], label='Input Neurons')
    h_scatter = plt.scatter(xMI[20:], yMI[20:], label='Hidden Neurons')
    plt.legend(handles=[x_scatter, h_scatter])
    plt.show()

    entropyList = loadfile('data/CS294Data/entropy_k1000_4.pkl')
    entropyList = np.array(entropyList)
    xMIN = xMI / entropyList
    yMIN = yMI / entropyList
    h_scatter = plt.scatter(xMIN[20:], yMIN[20:], label='Hidden Neurons')
    plt.legend(handles=[h_scatter])
    plt.show()

    #The following code yields non-useful results so I commented it.
    '''
    xyRatio = yMI/xMI
    ranking = loadfile('data/CS294Data/ranking_complex_1.pkl')
    ranking = loadfile('data/CS294Data/ranking_simple_removeNone_1.pkl')
    x_scatter = plt.scatter(ranking[:18], xyRatio[:18], label='Input Neurons')
    h_scatter = plt.scatter(ranking[20:], xyRatio[20:], label='Hidden Neurons')
    y_scatter = plt.scatter(ranking[18:20], xyRatio[18:20], label='Output Neurons')
    plt.legend(handles=[x_scatter, h_scatter, y_scatter])
    plt.show()
    '''

def plotStructure(cutOff = 0.05):
    infoTransfer = loadfile('data/CS294Data/transfer_k1000_6.pkl')
    infoTransfer = applySingleTimeMutualInformation(infoTransfer)
    #infoTransfer = normalizedInfoTransfer(infoTransfer, entropyList)
    list1 = transfersAbove(infoTransfer, cutOff, 0.0)
    #nodes = nodesInTransferList(list1)
    plotList(list1)

def analyzeRanking(rankingType):
    if rankingType == "simple":
        ranking = loadfile('data/CS294Data/ranking_simple_1.pkl')
        ranking = removeNone(ranking)
    else:
        ranking = loadfile('data/CS294Data/ranking_complex_1.pkl')

    savefile('data/CS294Data/ranking_simple_1.pkl', ranking)

    xrank = averageWithNone(ranking[:18])
    hrank = averageWithNone(ranking[20:])
    yrank = averageWithNone(ranking[18:20])
    print ("Average Input Neuron Rank: " + str(xrank))
    print ("Average Hidden Neuron Rank: " + str(hrank))
    print ("Average Output Neuron Rank: " + str(yrank))
    n, bins, patches = plt.hist(ranking[:18], 10, facecolor='green', alpha=0.75)
    n, bins, patches = plt.hist(ranking[20:], 10, facecolor='blue', alpha=0.75)
    n, bins, patches = plt.hist(ranking[18:20], 2, facecolor='red', alpha=0.75)
    plt.show()
    quit()

def MutualInformationDuringTraining(individualNeurons=False):
    numberOfEpochs = 1200
    allxMIOutputs = []
    allyMIOutputs = []
    allxMIHidden = []
    allyMIHidden = []
    a = 0
    while a < numberOfEpochs:
        name = 'data/CS294Data/trainValues' + str(a) + '.pkl'
        data = loadfile(name)
        if individualNeurons:
            (xMI, yMI) = mutualWithXY(data, noX=True)
            xMIOutputs = xMI[:2]
            yMIOutputs = yMI[:2]
            xMIHidden = xMI[2:]
            yMIHidden = yMI[2:]
        else:
            (hxMI, hyMI, oxMI, oyMI) = layerMutualWithXY(data)
            xMIOutputs = [oxMI]
            yMIOutputs = [oyMI]
            xMIHidden = [hxMI]
            yMIHidden = [hyMI]
        allxMIOutputs = allxMIOutputs + xMIOutputs
        allyMIOutputs = allyMIOutputs + yMIOutputs
        allxMIHidden = allxMIHidden + xMIHidden
        allyMIHidden = allyMIHidden + yMIHidden
        a += 1
    plt.scatter(allxMIHidden, allyMIHidden)
    plt.scatter(allxMIOutputs, allyMIOutputs)
    plt.show()

#The following is a list of all interesting results to run.
#analyzeRanking("complex")
#analyzeRanking("simple")
#plotStructure(cutOff=0.05)
#plotXYMutualInfo()

#The following will not work until the networks have been ran in all the training epoches.
#MutualInformationDuringTraining(individualNeurons=False)
