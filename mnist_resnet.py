import argparse
import os
import numpy as np

from cv2 import imwrite
from matplotlib import pyplot as plt
from tensorflow import cast, data, GradientTape, math
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.python.framework import dtypes
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import plot_model

"""
MNIST Classification with Resnet like architecture
"""

class Dataset:

    def __init__(self, *args, **kwargs):
        super(Dataset, self).__init__(*args, **kwargs)
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None
        self.x_test  = None
        self.y_test  = None
    
    @staticmethod
    def one_hot_encode(arr):
        if type(arr) != np.ndarray:
            raise ValueError("arr must be numpy array")
        arr_len = arr.shape[0]
        encoded_arr = np.zeros([arr_len, 10])
        for i in range(arr_len):
            encoded_arr[i, arr[i]] = 1
        return encoded_arr

    def prepare_dataset(self):
        try:

            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train, x_test = x_train / 255.0, x_test / 255.0 
            

            assert x_train.shape == (60000, 28, 28)
            assert x_test.shape == (10000, 28, 28)
            assert y_train.shape == (60000,)
            assert y_test.shape == (10000,)

        except AssertionError as e:
            raise Exception("Failed to load the dataset")
        
        self.x_train = x_train[:4000, :, :]
        self.y_train = self.one_hot_encode(y_train[:4000])
        self.x_valid = x_train[4000:5000, :, :]
        self.y_valid = self.one_hot_encode(y_train[4000:5000])
        self.x_test = x_test[:1000, :, :]
        self.y_test = self.one_hot_encode(y_test[:1000])


class NeuralNetwork:
    
    def __init__(self, *args, **kwargs):
        super(NeuralNetwork, self).__init__(*args, **kwargs)
        self.model = None


    def create_model(self):
        ip = layers.Input(shape=[784])
        repeat_vector = layers.RepeatVector(2)(ip)
        reshape = layers.Reshape(target_shape=[28, 28, -1])(repeat_vector)
        
        conv2d_1 = layers.Conv2D(filters=2, kernel_size=3,
                                 padding='same', activation='relu')(reshape)
        conv2d_2 = layers.Conv2D(filters=2, kernel_size=3,
                                 padding='same', activation='relu')(conv2d_1)
        concatenate_1 = layers.Concatenate()([reshape, conv2d_2])
        maxpool_1 = layers.MaxPool2D(pool_size=(2, 2),
                                     strides=(2, 2))(concatenate_1)
        
        conv2d_3 = layers.Conv2D(filters=4, kernel_size=3,
                                 padding='same', activation='relu')(maxpool_1)
        conv2d_4 = layers.Conv2D(filters=4, kernel_size=3,
                                 padding='same', activation='relu')(conv2d_3)
        concatenate_2 = layers.Concatenate()([maxpool_1, conv2d_4])
        maxpool_2 = layers.MaxPool2D(pool_size=(2, 2),
                                     strides=(2, 2))(concatenate_2)
        
        conv2d_5 = layers.Conv2D(filters=8, kernel_size=3,
                                 padding='same', activation='relu')(maxpool_2)
        conv2d_6 = layers.Conv2D(filters=8, kernel_size=3,
                                 padding='same', activation='relu')(conv2d_5)
        concatenate_3 = layers.Concatenate()([maxpool_2, conv2d_6])
        maxpool_3 = layers.MaxPool2D(pool_size=(2, 2),
                                     strides=(2, 2))(concatenate_3)
        
        conv2d_7 = layers.Conv2D(filters=16, kernel_size=3,
                                 padding='same', activation='relu')(maxpool_3)
        conv2d_8 = layers.Conv2D(filters=16, kernel_size=3,
                                 padding='same', activation='relu')(conv2d_7)
        concatenate_4 = layers.Concatenate()([maxpool_3, conv2d_8])
        maxpool_4 = layers.MaxPool2D(pool_size=(2, 2),
                                     strides=(2, 2))(concatenate_4)


        conv2d_9 = layers.Conv2D(filters=32, kernel_size=3,
                                 padding='same', activation='relu')(maxpool_4)
        conv2d_10 = layers.Conv2D(filters=32, kernel_size=3,
                                  padding='same', activation='relu')(conv2d_9)

        flatten = layers.Flatten()(conv2d_10)

        dense_1 = layers.Dense(20, activation="softmax")(flatten)
        dense_2 = layers.Dense(10, activation="softmax")(dense_1)

        self.model = Model(ip, dense_2)
    
    def print_summary(self):
        if not self.model:
            raise AssertionError("model is not initialized yet")
        print("-" * 80)
        print("**** Model Summary ****")
        print("-" * 80)
        print(self.model.summary())
        print("-" * 80)

    # def dump_model(self):
    #     plot_model(self.model, "model_architecture.png", show_shapes=True)
    #     with open("model_summary.txt", "w") as f:
    #         self.model.summary(print_fn=lambda x: f.write(x + "\n"))

        
    def train(self, dataset, optimizer, lossfn,
              batch_sz=100, learning_rate=0.005, epochs=150, metric=None):

        print("-" * 80)
        print("**** Training ****")
        print("-" * 80)

        if not self.model:
            raise AssertionError("Model is not initialized yet")
        
        if not metric:
            metric = Metrics()
        
        optimizer.learning_rate = learning_rate

        train_acc_metric = CategoricalAccuracy()
        val_acc_metric = CategoricalAccuracy()
        
        train_dataset = data.Dataset.from_tensor_slices((dataset.x_train,
                                                         dataset.y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_sz)

        val_dataset = data.Dataset.from_tensor_slices((dataset.x_valid,
                                                       dataset.y_valid))
        val_dataset = val_dataset.batch(batch_sz)

        train_accuracies = []
        train_losses = []
        validation_accuracies = []
        validation_losses = [] 

        for epoch in range(epochs):
            print("\nEpoch = {}".format(epoch+1))

            train_loss = 0
            val_loss = 0

            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                with GradientTape() as tape:
                    logits = self.model(x_batch_train, training=True)
                    loss = lossfn(y_batch_train, logits)
                    train_loss += loss
                grads = tape.gradient(loss, self.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                train_acc_metric.update_state(y_batch_train, logits)

            train_acc = float(train_acc_metric.result()) * 100
            train_loss = float(train_loss) / step

            print("Training Accuracy: {:.2f}".format(train_acc))
            print("Training loss: {:.4f}".format(train_loss))

            train_accuracies.append(train_acc)
            train_losses.append(train_loss)
            train_acc_metric.reset_states()

            for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
                val_logits = self.model(x_batch_val, training=False)
                val_loss += lossfn(y_batch_val, val_logits)
                val_acc_metric.update_state(y_batch_val, val_logits)
      
            val_acc = float(val_acc_metric.result()) * 100
            val_loss = float(val_loss) / step

            print("Validation Accuracy: {:.2f}".format(val_acc))
            print("Validation Loss: {:.4f}".format(val_loss))
      
            validation_accuracies.append(val_acc)
            validation_losses.append(val_loss)
            val_acc_metric.reset_states()

        metric.train_acc = train_accuracies
        metric.val_acc = validation_accuracies
        metric.train_loss = train_losses
        metric.val_loss = validation_losses
        metric.epochs = epochs
        return metric


    def test(self, dataset, metric=None):
        print("-" * 80)
        print("**** Test ****")
        print("-" * 80)

        if not self.model:
            raise AssertionError("Model is not initialized yet")

        if not metric:
            metric = Metrics()

        test_acc_metric = CategoricalAccuracy()
        logits = self.model(dataset.x_test, training=False)
        test_acc_metric.update_state(dataset.y_test, logits)
        test_acc = float(test_acc_metric.result()) * 100
        print("Test Accuracy: {:.2f}".format(test_acc))

        metric.test_acc = test_acc
        return metric

class Metrics:

    def __init__(self):
        super(Metrics, self).__init__()
        self.train_acc = None
        self.val_acc = None
        self.train_loss = None
        self.val_loss = None
        self.test_acc = None
        self.test_loss = None
        self.epochs = None
    
    @staticmethod
    def log_loss(labels, predictions):
        labels = cast(labels, dtype=dtypes.float32)
        predictions = cast(predictions, dtype=dtypes.float32)
        loss = math.multiply(- (1./labels.shape[0]),
                                math.reduce_sum(
                                    math.multiply(labels,
                                    math.log(predictions))))
        return loss

    @staticmethod
    def smooth(values, weight):
        prev = values[0]
        smooth_values = []
        for val in values:
            smoothed_val = prev * weight + (1 - weight) * val
            smooth_values.append(smoothed_val)
            prev = smoothed_val
        return smooth_values
    
    def plot_accuracy_graph(self, fpath):
        x = range(1, self.epochs+1)
        plt.title("Resnet-4 Accuracy")
        plt.xlabel("epoch")
        plt.ylabel("accuracy's")
        plt.plot(x, self.smooth(self.train_acc, 0.9),
                 label="train", color="skyblue")
        plt.plot(x, self.smooth(self.val_acc, 0.9),
                 label="valid", color="orange")
        plt.plot(x, self.smooth(np.repeat(self.test_acc, self.epochs), 0.9),
                 label="test", color="purple")
        plt.grid()
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(fpath, "accuracy.png"))
        plt.clf()
    
    def plot_loss_graph(self, fpath):
        x = range(1, self.epochs+1)
        plt.title("Resnet-4 Loss")
        plt.xlabel("epoch")
        plt.ylabel("loss's")
        plt.plot(x, self.smooth(self.train_loss, 0.9),
                 label="train", color="skyblue")
        plt.plot(x, self.smooth(self.val_loss, 0.9),
                 label="valid", color="orange")
        plt.grid()
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(fpath, "loss.png"))
        plt.clf()


def create_op_directory():
    try:
        opdir = os.path.join(os.path.abspath(os.path.join(__file__, os.pardir)),
                             "output")
        if not os.path.isdir(opdir):
            os.mkdir(opdir)
    except Exception as e:
        raise Exception("Failed to create output directory")
    return opdir

def dump_images(fpath, dataset, limit, train=True,
                test=True, validation=True):
    if not fpath:
        raise ValueError("fpath must be valid directory path")
    if not dataset:
        raise ValueError("dataset must be valid Dataset object")

    if not limit:
        train_limit, valid_limit, test_limit = len(dataset.x_train), \
            len(dataset.x_valid), len(dataset.x_test)
    else:
        train_limit, valid_limit, test_limit = limit, limit, limit

    if train:
        for i in range(train_limit):
            fname = os.path.join(fpath, "train_{}.png".format(i))
            imwrite(fname, dataset.x_train[i]*255)

    
    if test:
        for i in range(test_limit):
            fname = os.path.join(fpath, "test_{}.png".format(i))
            imwrite(fname, dataset.x_test[i]*255)


    if validation:
        for i in range(valid_limit):
            fname = os.path.join(fpath, "validation_{}.png".format(i))
            imwrite(fname, dataset.x_valid[i]*255)



def main(*args, **kwargs):
    opdir = create_op_directory()

    dataset = Dataset()
    dataset.prepare_dataset()

    dump_images(fpath=opdir, dataset=dataset, limit=5)

    nn = NeuralNetwork()
    nn.create_model()
    nn.print_summary()

    # uncomment dump_model function and relevant import
    # nn.dump_model()

    dataset.x_train = np.reshape(dataset.x_train, (-1, 784))
    dataset.x_valid = np.reshape(dataset.x_valid, (-1, 784))
    dataset.x_test = np.reshape(dataset.x_test, (-1, 784))

    metric = nn.train(dataset=dataset, optimizer=Adam(), 
                      lossfn=Metrics.log_loss, 
                      learning_rate=kwargs["learning_rate"],
                      epochs=kwargs["epochs"])
    
    nn.test(dataset=dataset, metric=metric)

    metric.plot_accuracy_graph(opdir)
    metric.plot_loss_graph(opdir)
    print("Metric graphs plotted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=150, type=int,
                        help="Number of epochs to run for")
    parser.add_argument("--learning_rate", default=0.005, type=float,
                        help="Learning rate to use for")
    args = parser.parse_args()
    main(learning_rate=args.learning_rate, epochs=args.epochs)
