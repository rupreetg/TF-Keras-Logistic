class ModelParameters(object): 

    def __init__(self, _input_feature_count, _layers, _activations, _output_classes):
        assert len(_layers) == len(_activations), "Number of layers should match number of activations"
        self.inputfeaturecount = _input_feature_count
        #TO DO ASSERT if len (layers) <> len(activation)
        self.layers = _layers
        self.activations = _activations
        self.outputclasses = _output_classes