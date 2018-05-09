from framework.models import Sequential
from framework.activations import ReLU, Sigmoid
from framework.layers import Linear
from framework.losses import MSE
from framework.optimizers import SGD, Adam
from framework import FloatTensor as Tensor
import matplotlib.pyplot as plt
import utilites
from data import loadDataSet


if __name__ == "__main__":
    plt.ion()

    data = loadDataSet(one_hot=False)

    # Create simple linear calssifier model
    lin1 = Linear(2, 10)
    relu1 = ReLU()
    lin2 = Linear(10, 1)
    sig1 = Sigmoid()
    model = Sequential([lin1, relu1, lin2, sig1])
    
    loss_algo = MSE()
    # optimizer_algo = SGD(eta=2e-2, lamb=0.2)
    optimizer_algo = Adam()

    # Training model
    epochs_per_step = 1
    for e in range(0, 100, epochs_per_step):
        # Testing model on training data
        pred_labels = model.predict(x=Tensor(data.test.data)).round().int()

        # Printing test accuracy
        acc = utilites.getAccuracy(data.test.labels.int(), pred_labels)
        print("Test Accuracy = {0:.2f}%".format(acc*100))

        # PLoting
        utilites.plot2Dset(data.test.data, data.test.labels.int(), pred_labels)
        # utilites.plotLinearSeperator(lin1.w.value, lin1.b.value)
        plt.pause(0.1)

        # Training model
        model.train(x=data.train.data,
                    y=data.train.labels,
                    batch_size=16,
                    optimizer=optimizer_algo,
                    loss=loss_algo,
                    epochs=epochs_per_step)
