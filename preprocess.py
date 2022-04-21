from libraries import *
from move import move

class preprocess:
    def __init__(self):
        self.minAll = 999999 #To determine min magnitude of distance between landmarks
        self.maxAll =  0 #To determine max magnitude of distance between landmarks

    def measureDiag(self, hand): #To recognize hand how much far away from camera
        hand = pd.DataFrame(hand, index = [i for i in range(hand.__len__())])
        minX, minY = hand.iloc[:, 1].min(axis = 0), hand.iloc[:, 2].min(axis = 0)
        maxX, maxY = hand.iloc[:, 1].max(axis = 0), hand.iloc[:, 2].max(axis = 0)

        self.diagValues = self.measureDist([0, minX, minY], [1, maxX, maxY])
        return self.measureDist([0, minX, minY], [1, maxX, maxY])

    def measureDist(self, hand1, hand2): #Find a vector length which is given between two points
        distance = float(math.sqrt(((abs(hand2[2] - hand1[2])) * (abs(hand2[2] - hand1[2]))) +
                               ((abs(hand2[1] - hand1[1]) * abs(hand2[1] - hand1[1]) ))))
        return distance

    def handFeatureExtracting(self, tag, hand): #Create a move which have landmarks distances and tag to give machine learning model
        move1 = move(tag, None)

        index1 = 0
        index2 = 1
        distance = dict()
        diagLength = self.measureDiag(hand)
        if diagLength < self.minAll:
            self.minAll = diagLength
        elif diagLength > self.maxAll:
            self.maxAll = diagLength

        while(index1 != hand.__len__() - 1):
            while (index2 != hand.__len__() - 1):
                value = self.measureDist(hand[index1], hand[index2])
                distance[f"{hand[index1][0]}_{hand[index2][0]}"] = value/diagLength #Normalizing with diagLength to preserve from increasing difference between close and far hand
                index2 += 1
            index1 += 1
            index2 = index1 + 1

        move1.handDistance = pd.DataFrame(distance, index = [0])

        return move1

    def handMovesTable(self, movesList = list): #Create table with landmarks distance and tags [191,HandsNumber]
        table = pd.DataFrame()
        tags = []
        for i in movesList:
            table = pd.concat([table, i.handDistance], axis = 0)
            tags.append(i.tag)

        table["tag"] = tags
        table.index = [i for i in range(table.__len__())]
        table = table.sample(frac = 1).reset_index(drop=True) #Shuffle the data
        return table

    def encodingBinary(self, y, targets):
        new_y = []
        for i in y:
            newTarget = []
            encodeValue = list(targets).index(str(i))
            for j in range(targets.__len__()):
                if j == encodeValue:
                    newTarget.append(1)
                else:
                    newTarget.append(0)
            new_y.append(newTarget)
        return pd.DataFrame(new_y)

    def flat(self, y = pd.DataFrame):
        flatted = []
        for i in range(y.iloc[:,0].__len__()):
            for j in range(y.iloc[0, :].__len__()):
                flatted.append(y.iloc[i, j])
        return flatted
