from torch import cat, FloatTensor


class Sequential(object):
    """
    Creates a network from a model from a list a modules.

    Parameters
    ----------
    layers : list
        A list of the different modules of the model. The dimention of the
        output of each layer must match the dimention of the input of the next
        layer.
    """
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        self.output = x
        return self.output

    def backward(self, gradwrtoutput):
        y = gradwrtoutput
        for l in reversed(self.layers):
            y = l.backward(y)

    def get_layers(self):
        return self.layers

    def train(self, x, y, optimizer, loss, batch_size=1, epochs=1,
              verbose=True):

        nb_samples = x.shape[0]

        # Casting into 2D tensor
        if y.dim() == 1:
            y = y.unsqueeze(1)

        # Extract batches
        batches_x = x.split(batch_size, dim=0)
        batches_y = y.split(batch_size, dim=0)

        # Training
        for e in range(epochs):
            total_loss = 0
            accuracy = 0
            if verbose:
                print("\nEpoch {0:d}/{1:d}".format(e+1, epochs))

            for batch_x, batch_y in zip(batches_x, batches_y):
                # Forward Pass
                output = self.forward(batch_x)
                grad = loss.dloss(output, batch_y)

                # Backward Pass
                self.backward(grad)
                optimizer.apply(self.layers)

                # Update tracking metric
                if verbose:
                    total_loss += sum(loss.loss(output, batch_y))

            # Display message
            if verbose:
                average_loss = total_loss/nb_samples
                print("Loss = {0:.2e}".format(average_loss, accuracy*100))

    def predict(self, x, batch_size=None):
        # Prep Ouput
        output = FloatTensor()

        # Casting into 2D tensor if required
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Extracting batches
        if batch_size is None:
            batches = [x]
        else:
            batches = x.split(batch_size, dim=0)

        # Propagating batches
        for batch in batches:
            # Forward Pass
            output = cat((output, self.forward(batch)), dim=0)

        return output.squeeze()
