from torch import FloatTensor, sigmoid, tanh, max


class Activations(object):
    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def get_param(self):
        raise NotImplementedError


class ReLU(Activations):
    """
    Rectified Linear Unit module. Performes the operation max(0, x) on the
    input tensor in an element-wise manner.
    """
    def forward(self, input):
        assert input.dim() == 2
        self.input = input
        return max(input, FloatTensor([0]))

    def backward(self, gradwrtoutput):
        assert gradwrtoutput.dim() == 2
        return (self.input > 0).float() * gradwrtoutput

    def get_param(self):
        return []


class Tanh(Activations):
    """
    Hyperbolic tangeant module. Performes the tanh(x) operation on the input
    tennor in element-wise manner.
    """
    def forward(self, input):
        assert input.dim() == 2
        self.input = input
        return tanh(input)

    def backward(self, gradwrtoutput):
        assert gradwrtoutput.dim() == 2
        return (1 - tanh(self.input)**2) * gradwrtoutput

    def get_param(self):
        return []


class Sigmoid(Activations):
    """
    Hyperbolic tangeant module. Performes the tanh(x) operation on the input
    tennor in element-wise manner.
    """
    def forward(self, input):
        assert input.dim() == 2
        self.input = input
        self.output = sigmoid(input)
        return self.output

    def backward(self, gradwrtoutput):
        assert gradwrtoutput.dim() == 2
        # print("ds = ", self.output * (1 - self.output) * gradwrtoutput)
        return self.output * (1 - self.output) * gradwrtoutput

    def get_param(self):
        return []
