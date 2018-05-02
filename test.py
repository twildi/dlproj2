from framework.models import Sequential
from framework.activations import ReLU, Sigmoid
from framework.layers import Linear
from framework.losses import MSE
from framework.optimizers import SGD
from framework import FloatTensor as Tensor
import matplotlib.pyplot as plt
import utilites
from data import loadDataSet


if __name__ == "__main__":
    plt.ion()
    simple = False

    data = loadDataSet(one_hot=False)

    # Create simple linear calssifier model
    lin1 = Linear(2, 10)
    relu1 = ReLU()
    lin2 = Linear(10, 10)
    relu2 = ReLU()
    lin3 = Linear(10, 1)
    sig1 = Sigmoid()
    model = Sequential([lin1, relu1, lin2, relu2, lin3, sig1])

    # Training model
    epochs_per_step = 1
    for e in range(0, 100, epochs_per_step):
        # Testing model on training data
        pred_labels = model.predict(x=Tensor(data.test.data))
        pred_labels = list(map(lambda x: int(x), pred_labels.round()))

        # Printing test accuracy
        acc = utilites.getAccuracy(data.test.labels, pred_labels)
        print("Test Accuracy = {0:.2f}%".format(acc*100))

        # PLoting
        utilites.plot2Dset(data.test.data, data.test.labels, pred_labels)
        # utilites.plotLinearSeperator(lin1.w.value, lin1.b.value)
        plt.pause(0.1)

        # Training model
        model.train(x=Tensor(data.train.data),
                    y=Tensor(data.train.labels),
                    batch_size=32,
                    optimizer=SGD(eta=2e-2),
                    loss=MSE(),
                    epochs=epochs_per_step)
