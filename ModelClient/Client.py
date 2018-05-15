



class ModelClient(object):
    def __init__(self, X, Y, Tag, LossFunction):
        self._X = X
        self._Y = Y
        self._Tag = Tag
        self._LossFunction = LossFunction

    def RunExpirement(self, X, Y, Tag, LossFunction):
        #TODO call model selector
        return
    