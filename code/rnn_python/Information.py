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

def normalize(data):
    total = 0.0
    a = 0
    while a < len(data):
        norm = np.sum(np.array(data[a]) ** 2)
        total += norm
        a += 1
    avg = total / float(len(data) * len(data[0]))
    data = list(np.array(data) / avg)
    return data

def maxNormalize(data):
    data = np.array(data)
    data = list(data / np.max(data))
    return data

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

def calculateInformationTransfer(data, cutExamples="No"):
    infoTransfer = []
    numberOfNodes = len(data)
    #examplesToRunOn = 1000 #should be len(data[0][0]) for the actual final results
    if cutExamples == "No":
        examplesToRunOn = len(data[0][0])
    else:
        examplesToRunOn = cutExamples
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
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1) #
    pos = nx.spring_layout(G)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size = 500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, arrows=True)
    print ("nodes:" + str(len(np.unique(list3))))
    plt.title("The Emergent Structure", fontsize=10)
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

        yData = loadfile('data/CS294Data/yData_testing.pkl')
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

        yData = loadfile('data/CS294Data/yData_testing.pkl')
        #Input, Hidden, Output, True Y Values.
        return (xData, hData, oData, yData)

    (xData, hData, oData, yData) = processData(data)
    examplesToRunOn = len(hData)
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
    infoTransfer = loadfile('data/CS294Data/transfer.pkl')
    infoTransfer = applySingleTimeMutualInformation(infoTransfer)

    list1 = transfersAbove(infoTransfer, 0.3, 0.0)
    ranking1 = rankingAlgorithm(list1)
    savefile('data/CS294Data/ranking_simple1.pkl', ranking1)
    list2 = transfersAbove(infoTransfer, 0.4, 0.0)
    ranking2 = rankingAlgorithm(list2)
    savefile('data/CS294Data/ranking_simple2.pkl', ranking2)

    list3 = transfersAbove(infoTransfer, 0.06, 0.0)
    cranking = complexRankingAlgorithm(infoTransfer)
    savefile('data/CS294Data/ranking_complex.pkl', cranking)
    sranking = rankingAlgorithm(list3)
    savefile('data/CS294Data/ranking_simple.pkl', sranking)
    nrsranking = removeNone(sranking)
    savefile('data/CS294Data/ranking_simple_removeNone.pkl', nrsranking)


def saveTransferAndEntropy():
    data1 = loadfile('data/CS294Data/CON_fullyTrained.pkl')
    #data1 = loadfile('data/CS294Data/testValues1.pkl')
    data2 = makeDataEqual(data1)
    data2 = giveClippedAndDelayed(data2)
    data2 = np.array(data2)
    (infoTransfer, entropyList) = calculateInformationTransfer(data2)
    output = open('data/CS294Data/transfer.pkl', 'wb')
    pickle.dump(infoTransfer, output)
    output.close()
    output = open('data/CS294Data/entropy.pkl', 'wb')
    pickle.dump(entropyList, output)
    output.close()

def saveTestingYData():
    data = loadfile('data/CS294Data/CON_fullyTrained.pkl')
    yData = []
    a = 0
    while a < len(data):
        b = 0
        while b < len(data[a][0]):
            yData.append(data[a][2][b])
            b += 1
        a += 1
    savefile('data/CS294Data/yData_testing.pkl', yData)

def MutualInformationDuringTraining(individualNeurons=False):
    allxMIOutputs = []
    allyMIOutputs = []
    allxMIHidden = []
    allyMIHidden = []
    bigData = loadfile('data/CS294Data/CON.pkl')
    a = 0
    while a < len(bigData):
        data = bigData[a]
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

    return (allxMIHidden, allyMIHidden, allxMIOutputs, allyMIOutputs)

def errorDuringTraining():
    def getOutput(data):
        oData = []
        a = 0
        while a < len(data):
            b = 0
            while b < len(data[a][0]):
                oData.append(data[a][2][b])
                b += 1
            a += 1
        return oData

    yData = loadfile('data/CS294Data/yData_testing.pkl')
    size = np.array(yData).size
    errors = []
    bigData = loadfile('data/CS294Data/CON.pkl')
    a = 0
    while a < len(bigData):
        data = bigData[a]
        oData = getOutput(data)
        error = np.sum(np.abs(np.array(oData) - np.array(yData)))
        avgError = error / float(size)
        errors.append(avgError)
        a += 1
    return errors

def calculateComplexity(variables):
    def averageSubsetEntropy(variables, subsets):
        totalEntropy = 0.0
        a = 0
        while a < len(subsets):
            subset = np.array(subsets[a])
            vars1 = np.array(variables)[subset]
            vars1 = list(vars1.T)
            entropy = ee.entropy(vars1, k=10)
            totalEntropy += entropy
            a += 1
        totalEntropy = totalEntropy / float(len(subsets))
        return totalEntropy

    def randomSubsets(setsize, subsetsize):
        subsets = []
        numberOfSubsets = 5
        a = 0
        while a < numberOfSubsets:
            choice1 = np.random.choice(setsize, subsetsize, replace=False)
            subsets.append(list(choice1))
            a += 1
        return subsets

    def allAverageSubsetEntropies(variables):
        entropies = []
        a = 1
        while a < len(variables):
            subsets = randomSubsets(len(variables), a)
            entropy1 = averageSubsetEntropy(variables, subsets)
            entropies.append(entropy1)
            a += 1
        bigSetEntropy = ee.entropy(list(np.array(variables).T), k=10)
        return (entropies, bigSetEntropy)

    (entropies, bigSetEntropy) = allAverageSubsetEntropies(variables)
    size1 = len(variables)
    coefficient = float(size1 - 1) / 2.0
    complexity = sum(entropies) - (coefficient * bigSetEntropy)
    return complexity

def calculateTrainingComplexity():
    def processData(data):
        allData = []
        a = 0
        while a < len(data):
            b = 0
            while b < len(data[a][0]):
                allData.append(list(data[a][0][b]) + list(data[a][2][b]) + list(data[a][1][b]))
                b += 1
            a += 1

        allData = list(np.array(allData).T)
        return allData

    complexities = []
    bigData = loadfile('data/CS294Data/CON.pkl')
    a = 0
    while a < len(bigData):
        print (a)
        data = bigData[a]
        allData = processData(data)
        complexity = calculateComplexity(allData)
        complexities.append(complexity)
        a += 1
    return complexities

def saveErrorPredictors():
    print ("2")
    complexities = calculateTrainingComplexity()
    name2 = 'data/CS294Data/trainingComplexities1.pkl'
    savefile(name2, complexities)
    print ("2")
    errors = errorDuringTraining()
    name3 = 'data/CS294Data/trainingErrors1.pkl'
    savefile(name3, errors)
    print ("3")

#################################################
###                                           ###
###       Functions for Final Processes       ###
###                                           ###
#################################################

def plotXYMutualInfo():
    data1 = loadfile('data/CS294Data/CON_fullyTrained.pkl')
    (xMI, yMI) = mutualWithXY(data1)
    xMI = np.array(xMI)
    yMI = np.array(yMI)

    x_scatter = plt.scatter(xMI[:18], yMI[:18], label='Input Neurons')
    h_scatter = plt.scatter(xMI[20:], yMI[20:], label='Hidden Neurons')
    y_scatter = plt.scatter(xMI[18:20], yMI[18:20], label='Output Neurons')
    plt.legend(handles=[x_scatter, h_scatter, y_scatter])
    plt.title('Input Mutual Information Versus Output Mutual Information')
    plt.xlabel("Mutual Information With Input")
    plt.ylabel("Mutual Information With Output")
    plt.show()

    x_scatter = plt.scatter(xMI[:18], yMI[:18], label='Input Neurons')
    h_scatter = plt.scatter(xMI[20:], yMI[20:], label='Hidden Neurons')
    plt.legend(handles=[x_scatter, h_scatter])
    plt.title('Input Mutual Information Versus Output Mutual Information')
    plt.xlabel("Mutual Information With Input")
    plt.ylabel("Mutual Information With Output")
    plt.show()

    entropyList = loadfile('data/CS294Data/entropy.pkl')
    entropyList = np.array(entropyList)
    xMIN = xMI / entropyList
    yMIN = yMI / entropyList
    h_scatter = plt.scatter(xMIN[20:], yMIN[20:], label='Hidden Neurons')
    plt.legend(handles=[h_scatter])
    plt.xlabel("Mutual Information With Input")
    plt.ylabel("Mutual Information With Output")
    plt.title('Input Mutual Information Versus Output Mutual Information (normalized using entropy)')
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
    infoTransfer = loadfile('data/CS294Data/transfer.pkl')
    infoTransfer = applySingleTimeMutualInformation(infoTransfer)
    #infoTransfer = normalizedInfoTransfer(infoTransfer, entropyList)
    list1 = transfersAbove(infoTransfer, cutOff, 0.0)
    #nodes = nodesInTransferList(list1)
    plotList(list1)

def analyzeRanking(rankingType, cutoffMode=0):
    if rankingType == "simple":
        if cutoffMode == 0:
            ranking = loadfile('data/CS294Data/ranking_simple.pkl')
        elif cutoffMode == 1:
            ranking = loadfile('data/CS294Data/ranking_simple1.pkl')
        elif cutoffMode == 2:
            ranking = loadfile('data/CS294Data/ranking_simple2.pkl')
        ranking = removeNone(ranking)
    else:
        ranking = loadfile('data/CS294Data/ranking_complex.pkl')

    xrank = averageWithNone(ranking[:18])
    hrank = averageWithNone(ranking[20:])
    yrank = averageWithNone(ranking[18:20])
    print ("Average Input Neuron Rank: " + str(xrank))
    print ("Average Hidden Neuron Rank: " + str(hrank))
    print ("Average Output Neuron Rank: " + str(yrank))
    multiClassHistogram(ranking[:18], ranking[20:], ranking[18:20])
    x = [ranking[:18], ranking[20:], ranking[18:20]]
    with open('rank2.csv', 'w') as f:
        for i in range(len(x[0])):
            f.write(str(x[0][i]) + ',' + 'input' + '\n')
        for i in range(len(x[1])):
            f.write(str(x[1][i]) + ',' + 'hidden' + '\n')
        for i in range(len(x[2])):
            f.write(str(x[2][i]) + ',' + 'output' + '\n')

    #n, bins, patches = plt.hist(ranking[:18], 10, facecolor='green', alpha=0.75)
    #n, bins, patches = plt.hist(ranking[20:], 10, facecolor='blue', alpha=0.75)
    #n, bins, patches = plt.hist(ranking[18:20], 2, facecolor='red', alpha=0.75)
    #plt.show()

def saveTrainingMI():
    (allxMIHidden, allyMIHidden, allxMIOutputs, allyMIOutputs) = MutualInformationDuringTraining(individualNeurons=False)
    data1 = [allxMIHidden, allyMIHidden, allxMIOutputs, allyMIOutputs]
    savefile('data/CS294Data/TrainingMI_Layer2.pkl', data1)

    (allxMIHidden, allyMIHidden, allxMIOutputs, allyMIOutputs) = MutualInformationDuringTraining(individualNeurons=True)
    data2 = [allxMIHidden, allyMIHidden, allxMIOutputs, allyMIOutputs]
    savefile('data/CS294Data/TrainingMI_Individual2.pkl', data2)

def plotTrainingMI(individualNeurons=False):
    if individualNeurons:
        data = loadfile('data/CS294Data/TrainingMI_Individual.pkl')
    else:
        data = loadfile('data/CS294Data/TrainingMI_Layer.pkl')
    (allxMIHidden, allyMIHidden, allxMIOutputs, allyMIOutputs) = (data[0], data[1], data[2], data[3])
    allxMIHidden = maxNormalize(allxMIHidden)
    allyMIHidden = maxNormalize(allyMIHidden)
    allxMIOutputs = maxNormalize(allxMIOutputs)
    allyMIOutputs = maxNormalize(allyMIOutputs)

    if individualNeurons:
        plt.scatter(allxMIHidden, allyMIHidden)
        plt.title("Training Mutual Information Of The Hidden Layer")
        plt.xlabel("Mutual Information With The Input")
        plt.ylabel("Mutual Information With The Output")
        plt.show()

        plt.scatter(allxMIOutputs, allyMIOutputs)
        plt.title("Training Mutual Information Of The Output Layer")
        plt.xlabel("Mutual Information With The Input")
        plt.ylabel("Mutual Information With The Output")
        plt.show()
    else:
        plt.plot(allxMIHidden, allyMIHidden)
        plt.title("Training Mutual Information Of The Hidden Layer")
        plt.xlabel("Mutual Information With The Input")
        plt.ylabel("Mutual Information With The Output")
        plt.show()

        plt.plot(allxMIOutputs, allyMIOutputs)
        plt.title("Training Mutual Information Of The Output Layer")
        plt.xlabel("Mutual Information With The Input")
        plt.ylabel("Mutual Information With The Output")
        plt.show()

def plotErrorPredictors():
    def minipPlotErrorPredictors(mode=0):
        data1 = loadfile('data/CS294Data/TrainingMI_Layer.pkl')
        allxMIHidden = data1[0]
        allyMIHidden = data1[1]
        allxMIOutputs = data1[2]
        allyMIOutputs = data1[3]

        complexities = loadfile('data/CS294Data/trainingComplexities1.pkl')
        errors = loadfile('data/CS294Data/trainingErrors1.pkl')
        def fitOnErrorPlot(data):
            data = np.array(data)
            data = data - np.min(data)
            data = data / (np.max(data))
            return data

        allxMIHidden = fitOnErrorPlot(allxMIHidden)
        allyMIHidden = fitOnErrorPlot(allyMIHidden)
        allxMIOutputs = fitOnErrorPlot(allxMIOutputs)
        allyMIOutputs = fitOnErrorPlot(allyMIOutputs)
        complexities = fitOnErrorPlot(complexities)
        errors = fitOnErrorPlot(errors)
        epochs = np.arange(len(complexities)) * 10
        if mode == 0:
            xhPlot, = plt.plot(epochs, allxMIHidden, label="Input, Hidden Layer Mutual Information")
            yhPlot, = plt.plot(epochs, allyMIHidden, label="Correct Output, Hidden Layer Mutual Information")
            xoPlot, = plt.plot(epochs, allxMIOutputs, label="Input, Output Layer Mutual Information")
            yoPlot, = plt.plot(epochs, allyMIOutputs, label="Correct Output, Output Layer Mutual Information")
            cxPlot, = plt.plot(epochs, complexities, label="Complexity")
            errorPlot, = plt.plot(epochs, errors, label="Error")
            plt.legend(handles=[xhPlot, yhPlot, xoPlot, yoPlot, cxPlot, errorPlot])
        elif mode == 1:
            cxPlot, = plt.plot(epochs, complexities, label="Complexity")
            errorPlot, = plt.plot(epochs, errors, label="Error")
            plt.legend(handles=[cxPlot, errorPlot])
        elif mode == 2:
            xhPlot, = plt.plot(epochs, allxMIHidden, label="Input, Hidden Layer Mutual Information")
            xoPlot, = plt.plot(epochs, allxMIOutputs, label="Input, Output Layer Mutual Information")
            cxPlot, = plt.plot(epochs, complexities, label="Complexity")
            errorPlot, = plt.plot(epochs, errors, label="Error")
            plt.legend(handles=[xhPlot, xoPlot, cxPlot, errorPlot])
        elif mode == 3:
            yoPlot, = plt.plot(epochs, allyMIOutputs, label="Correct Output, Output Layer Mutual Information")
            errorPlot, = plt.plot(epochs, errors, label="Error")
            plt.legend(handles=[yoPlot, errorPlot])
        elif mode == 4:
            yhPlot, = plt.plot(epochs, allyMIHidden, label="Correct Output, Hidden Layer Mutual Information")
            errorPlot, = plt.plot(epochs, errors, label="Error")
            plt.legend(handles=[yhPlot, errorPlot])
        elif mode == 5:
            xhPlot, = plt.plot(epochs, allxMIHidden, label="Input, Hidden Layer Mutual Information")
            xoPlot, = plt.plot(epochs, allxMIOutputs, label="Input, Output Layer Mutual Information")
            errorPlot, = plt.plot(epochs, errors, label="Error")
            plt.legend(handles=[xhPlot, xoPlot, errorPlot])
        #plt.plot(errors)
        plt.ylabel("Scaled Variables")
        plt.xlabel("Epoch")
        plt.show()

    minipPlotErrorPredictors(mode=0)
    minipPlotErrorPredictors(mode=1)
    minipPlotErrorPredictors(mode=2)
    minipPlotErrorPredictors(mode=3)
    minipPlotErrorPredictors(mode=4)
    minipPlotErrorPredictors(mode=5)

def shapePrinter(data):
    if (type(data) == list):
        print (len(data))
        shapePrinter(data[0])

def multiClassHistogram(data1, data2, data3):
    n_bins = 10
    x = [data1, data2, data3]

    fig, axes = plt.subplots(nrows=1, ncols=1)
    ax0 = axes

    colors = ['red', 'blue', 'lime']
    labels = ['Inputs', 'Hidden Neurons', 'Outputs']
    ax0.hist(x, n_bins, normed=1, histtype='bar', color=colors, label=labels)
    ax0.legend(prop={'size': 10})
    ax0.set_title('Neuron Rank Histogram')
    plt.xlabel("Rank")
    plt.ylabel("Scaled Number of Neurons")
    fig.tight_layout()
    plt.show()

def plotBasicHiddenChanging():
    #This just plots the rate of change of the hidden neuron activations
    bigData = loadfile('data/CS294Data/CON.pkl')
    changes = []
    a = 0
    while a < len(bigData):
        data = bigData[a]
        data = makeDataEqual(data)
        data = np.array(data)
        if a != 0:
            change = np.sum(np.abs(data - lastData))
            changes.append(change)
        lastData = data
        a += 1
    plt.plot(changes)
    plt.show()

def calcMeanVar(series):
    if type(series[0]) == list:
        mean1 = np.mean(series, axis=0)
        var1 = np.std(series, axis=0)
        #print (var1[:20])
        #print (mean1[:20])
        mean1 = np.linalg.norm(mean1)
        var1 = np.linalg.norm(var1)
        avg1 = 0.0
        a = 0
        while a < len(series):
            avg1 += np.linalg.norm(series[a])
            a += 1
        avg1 = avg1 / len(series)
    else:
        mean1 = abs(np.mean(series))
        var1 = abs(np.std(series))
        avg1 = 0.0
        a = 0
        while a < len(series):
            avg1 += abs(series[a])
            a += 1
        avg1 = avg1 / len(series)
    return (mean1, var1, avg1)

def calcSeriesMeanVar(series, spacing):
    means = []
    stds = []
    avgs = []
    miniSeries = []
    a = 0
    while a < len(series):
        miniSeries.append(series[a])
        if ((a + 1) % spacing) == 0:
            (mean1, var1, avg) = calcMeanVar(miniSeries)
            miniSeries = []
            means.append(mean1)
            stds.append(var1)
            avgs.append(avg)
        a += 1
    return (means, stds, avgs)

def convertGradSeries(data):
    data2 = []
    a = 0
    while a < len(data[0]):
        data2.append([])
        b = 0
        while b < len(data):
            data2[a].append(list(np.ndarray.flatten(np.array(data[b][a]))))
            b += 1
        a += 1
    return data2

def plotGradientMeanStd():
    def plotBasicMeanStd(data2, spacing):
        xAxis = np.arange(len(data2[0])/spacing) * 10
        (means1, stds1, avgs) = calcSeriesMeanVar(data2[1], spacing)
        (means2, stds2, avgs) = calcSeriesMeanVar(data2[2], spacing)
        (means3, stds3, avgs) = calcSeriesMeanVar(data2[3], spacing)
        means4 = (np.array(means1) + np.array(means2) + np.array(means3)) / 3.0
        stds4 = (np.array(stds1) + np.array(stds2) + np.array(stds3)) / 3.0
        means4 = means4 - means4[-1] #Changes zero point.
        meanplot, = plt.plot(xAxis, means4, label='Mean') #semilogx
        stdplot, = plt.plot(xAxis, stds4, label='Standard Devation') #semilogx
        #plt.plot(xAxis, means1)
        #plt.plot(xAxis, stds1)
        plt.legend(handles=[meanplot, stdplot])
        plt.xlabel("Iteration")
        plt.ylabel("Mean or Standard Devation")
        plt.title("Properties Of The Gradient During Training")
        plt.show()
    data = list(loadfile('data/CS294Data/CON_GRAD.pkl'))
    spacing = 10
    data2 = convertGradSeries(data)
    plotBasicMeanStd(data2, spacing)

def structureAnalysis(cutOff = 0.06):
    infoTransfer = loadfile('data/CS294Data/transfer.pkl')
    infoTransfer = applySingleTimeMutualInformation(infoTransfer)
    #infoTransfer = normalizedInfoTransfer(infoTransfer, entropyList)
    list1 = transfersAbove(infoTransfer, cutOff, 0.0)
    #nodes = nodesInTransferList(list1)
    degreesIn = list(np.zeros(45))
    degreesOut = list(np.zeros(45))
    a = 0
    while a < len(list1):
        degreesIn[list1[a][0]] += 1
        degreesOut[list1[a][1]] += 1
        a += 1
    #print (degreesIn)
    #print (degreesOut)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1) #
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    plt.scatter(degreesIn, degreesOut)
    plt.title("Input Degree VS Output Degree", fontsize=15)
    plt.xlabel("Input Degree", fontsize=12)
    plt.ylabel("Output Degree", fontsize=12)
    name = 'InOutDegreePlot' + str(cutOff) + '.png'
    # plt.savefig(name)
    plt.show()

    # n_bins = 10
    # x = [degreesIn, degreesOut]

    # fig, axes = plt.subplots(nrows=1, ncols=1)
    # ax0 = axes

    # colors = ['red', 'blue']
    # labels = ['Input Degree', 'Output Degree']
    # ax0.hist(x, n_bins, normed=1, histtype='bar', color=colors, label=labels)
    # print(x)
    # with open('test2.csv', 'w') as f:
    #     for i in range(len(x[0])):
    #         f.write(str(x[0][i]) + ',' + str(x[1][i]) + '\n')
    # ax0.legend(prop={'size': 15})
    # ax0.set_title('Neuron Degree Histogram', fontsize=20)
    # ax0.xaxis.set_tick_params(labelsize=15)
    # ax0.yaxis.set_tick_params(labelsize=15)

    # fig.tight_layout()
    # name = 'InOutDegreeHist' + str(cutOff) + '.png'
    # plt.ylabel("Percentage of Neurons", fontsize=15)
    # plt.xlabel("Degree", fontsize=15)
    # plt.savefig(name)
    # #plt.savefig(name)
    # plt.show()


def SeeStructure1(cutOff = 0.05):
    def intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    infoTransfer = loadfile('data/CS294Data/transfer.pkl')
    infoTransfer = applySingleTimeMutualInformation(infoTransfer)
    #infoTransfer = normalizedInfoTransfer(infoTransfer, entropyList)
    list1 = transfersAbove(infoTransfer, cutOff, 0.0)
    neuronList = []
    a = 0
    while a < 45:
        neuronList.append([[], []])
        a += 1
    a = 0
    while a < len(list1):
        neuronList[list1[a][0]][1].append(list1[a][1])
        neuronList[list1[a][1]][0].append(list1[a][0])
        a += 1
    compartmentSimilarity = []
    inSims = []
    outSims = []
    a1 = 0
    while a1 < 25:
        compartmentSimilarity.append([])
        b1 = 0
        while b1 < 25:
            if a1 != b1:
                a = a1 + 20
                b = b1 + 20
                inSim = len(intersection(neuronList[a][0], neuronList[b][0])) / (len(neuronList[a][0]) + 1)
                outSim = len(intersection(neuronList[a][1], neuronList[b][1])) / (len(neuronList[a][1]) + 1)
                compartmentSimilarity[a1].append([inSim, outSim])
                #if a == 20:
                inSims.append(inSim)
                outSims.append(outSim)
            b1 += 1
        a1 += 1

    #print (compartmentSimilarity)
    plt.scatter(inSims, outSims)
    plt.show()
    #nodes = nodesInTransferList(list1)
    #print (list1)
#These are just some basic, non-conclusive structure analysis results
# structureAnalysis(cutOff = 0.06)
# structureAnalysis(cutOff = 0.3)
# structureAnalysis(cutOff = 0.4)

#The following is a list of all interesting results to run.
# plotStructure(cutOff=0.06)
#plotStructure(cutOff=0.3)
#plotStructure(cutOff=0.4)
# analyzeRanking("simple", cutoffMode=0)
# analyzeRanking("simple", cutoffMode=1)
analyzeRanking("simple", cutoffMode=2)
#plotXYMutualInfo()

#plotTrainingMI(individualNeurons=False)
#plotTrainingMI(individualNeurons=True)
#plotErrorPredictors()
#plotGradientMeanStd()
