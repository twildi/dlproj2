from framework import Sequential, SGD, Sigmoid, Linear, MSE
from framework import FloatTensor as Tensor
import matplotlib.pyplot as plt
import utilites
from data import loadDataSet


if __name__ == "__main__":
    plt.ion()
    simple = False

    data = loadDataSet(one_hot=False)

    # Create simple linear calssifier model
    lin1 = Linear(2, 1, std=1)
    sig1 = Sigmoid()
    model = Sequential([lin1, sig1])

    # Training model
    epochs_per_step = 1
    for e in range(0, 10, epochs_per_step):
        # Testing model on training data
        pred_labels = model.predict(x=Tensor(data.test.data))
        pred_labels = list(map(lambda x: int(x), pred_labels.round()))

        # PLoting
        utilites.plot2Dset(data.test.data, data.test.labels, pred_labels)
        utilites.plotLinearSeperator(lin1.w.value, lin1.b.value)
        plt.pause(0.1)

        # Training model
        model.train(x=Tensor(data.train.data),
                    y=Tensor(data.train.labels),
                    batch_size=2,
                    optimizer=SGD(eta=2e-2),
                    loss=MSE(),
                    epochs=epochs_per_step)
