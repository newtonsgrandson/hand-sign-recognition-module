from libraries import *
import libraries

class scratchModel:
    def __init__(self):
        self.data = libraries.data
        self.pr = preprocess()

        self.X = self.data.iloc[:, 0:self.data.iloc[0].__len__() - 1]
        self.X = pd.concat([self.X, pd.Series(np.zeros(self.X.iloc[:, 0].__len__()), name="bias")], axis = 1)
        self.y = data.loc[:, "tag"]
        self.targets = self.y.unique()
        self.yEncode = self.pr.encodingBinary(self.y, self.targets)
        self.thetaT = list(np.zeros((self.X.shape[1])))

        print(self.costFunction(self.thetaT, self.X, self.y))

    def hypothesis(self, X, theta):
        z = np.dot(X, theta)
        return 1/(1 + np.exp(-z))

    def costFunction(self, theta = 0, X = pd.DataFrame, y = pd.Series):
        Axis_XLength = y.shape[0]
        yTheta = self.hypothesis(X, theta)
        yEncode = np.array(self.yEncode)
        print(yEncode * np.log(yTheta))
        return -(1/Axis_XLength) * np.sum(y*np.log(yTheta) + (1 - y)*np.log(1 - yTheta))

    def gradient(self, theta, X = pd.DataFrame, y = pd.Series):
        Axis_XLength = X.shape[0]
        yTheta = self.hypothesis(X, theta)
        return (1/Axis_XLength) * np.dot(X.transpose, yTheta - y)

    def fit(self, theta, X = pd.DataFrame, y = pd.DataFrame):
        opt_weigths = fmin_tnc(func = self.costFunction, x0 = theta,
                               fprime = self.gradient, args = (X, self.pr.flat(y)))
        return opt_weigths[0]

    def findParameters(self, theta, X = pd.DataFrame, y = pd.Series):
        yUnique = y.unique()
        thetaList = []
        for i in yUnique:
            yBinarize = self.pr.encodingBinary(y, self.targets)
            thetaI = self.fit(theta, X, yBinarize)
            thetaList.append(thetaI)
        return thetaList


scratchModel()
