import libraries
from libraries import *

class LRmodel: #Our model: Logistic Regression
    def __init__(self, random_state = 42):
        self.pr = preprocess() #Our preprocess class
        self.random_state_pos = random_state
        self.data = libraries.data

        X = self.data.iloc[:, 0:self.data.iloc[0].__len__() - 1] #Seperating X
        y = self.data.loc[:, "tag"] #Seperating Y

        self.model = LogisticRegression(random_state=self.random_state_pos) #Model
        self.model.fit(X, y) #Fitting

    def predict(self, hand):
        move = self.pr.handFeatureExtracting(hand = hand, tag = None) #Will be predicted move
        tag = self.model.predict(move.handDistance)[0] #Prediction
        move.tag = tag
        return move

def main():
    random_state_pos = 42
    data = pd.read_csv("data.csv", index_col=0) #Our data which we created at handDetectorModule

    X = data.iloc[:, 0:data.iloc[0].__len__()-1] #Our data X
    y = data.loc[:,"tag"] #Our data Y

    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=random_state_pos)
    model = LogisticRegression(random_state=random_state_pos)
    model.fit(train_X, train_y)
    predictions = model.predict(test_X)
    #To observe y_test and prediction
    test_y = pd.Series(test_y, name = "tagTest_y")
    predictions = pd.Series(predictions, index=test_y.index, name="tagPredictions")

    test_y = test_y.sort_index(axis = 0)
    predictions = predictions.sort_index(axis = 0)

    #If we want to calculate mae we should encode our target data
    encoder = LabelEncoder()
    encoder.fit(["a", "b", "c", "d"]) #target data
    encodeTest_Y = encoder.transform(test_y)
    encodePredictions = encoder.transform(predictions)

    maeFirst = mean_absolute_error(encodeTest_Y, encodePredictions)
    print(maeFirst) #You should see 0!
    table = pd.DataFrame([test_y, predictions])
    logic = [test_y == predictions]
    print(table)

if __name__ == "__main__":
    main()