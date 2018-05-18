from framework.models import Sequential
from framework.activations import ReLU, Sigmoid
from framework.layers import Linear
from framework.losses import MSE
from framework.optimizers import Adam  # SGD
import utilites
from data import loadDataSet

if __name__ == "__main__":
    data = loadDataSet(one_hot=True)

    # Create simple linear calssifier model
    lin1 = Linear(2, 25, std=0.1)
    relu1 = ReLU()
    lin2 = Linear(25, 25, std=0.1)
    relu2 = ReLU()
    lin3 = Linear(25, 25, std=0.1)
    relu3 = ReLU()
    lin4 = Linear(25, 2, std=0.1)
    sig1 = Sigmoid()
    model = Sequential([lin1, relu1, lin2, relu2, lin3, relu3, lin4, sig1])

    loss = MSE()
    optimizer = Adam()

    # Training model
    model.train(x=data.train.data,
                y=data.train.labels,
                batch_size=8,
                optimizer=optimizer,
                loss=loss,
                epochs=100)

    # Testing model on test data
    test_pred_labels = model.predict(x=data.test.data).round().int()
    train_pred_labels = model.predict(x=data.train.data).round().int()

    # Printing test accuracy
    test_acc = utilites.getAccuracy(data.test.labels.int(),
                                    test_pred_labels,
                                    one_hot=True)
    train_acc = utilites.getAccuracy(data.train.labels.int(),
                                     train_pred_labels,
                                     one_hot=True)
    print("Train Accuracy = {0:.2f}%, Test Accuracy = {1:.2f}%".
          format(train_acc*100, test_acc*100))
