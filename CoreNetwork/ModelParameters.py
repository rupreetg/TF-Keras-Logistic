class ModelParameters(object): 

    def __init__(self, _input_feature_count, _layers, _activations, _output_classes):
        assert len(_layers) == len(_activations), "Number of layers should match number of activation"
        self.input_feature_count = _input_feature_count
        self.layers = _layers
        self.activations = _activations
        self.output_classes = _output_classes