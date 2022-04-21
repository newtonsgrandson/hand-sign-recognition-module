from libraries import *

class move: #A move object with landmarks distance values and tag
    def __init__(self, tag = None, handDistance = None):
        self.tag = tag
        self.handDistance = handDistance