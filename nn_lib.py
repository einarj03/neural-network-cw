import numpy as np
import pickle


def xavier_init(size, gain=1.0):
    """
    Xavier initialization of network weights.
    """
    low = -gain * np.sqrt(6.0 / np.sum(size))
    high = gain * np.sqrt(6.0 / np.sum(size))
    return np.random.uniform(low=low, high=high, size=size)


class Layer:
    """
    Abstract layer class.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def update_params(self, *args, **kwargs):
        pass


class MSELossLayer(Layer):
    """
    MSELossLayer: Computes mean-squared error between y_pred and y_target.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def _mse(y_pred, y_target):
        return np.mean((y_pred - y_target) ** 2)

    @staticmethod
    def _mse_grad(y_pred, y_target):
        return 2 * (y_pred - y_target) / len(y_pred)

    def forward(self, y_pred, y_target):
        self._cache_current = y_pred, y_target
        return self._mse(y_pred, y_target)

    def backward(self):
        return self._mse_grad(*self._cache_current)


class CrossEntropyLossLayer(Layer):
    """
    CrossEntropyLossLayer: Computes the softmax followed by the negative log-
    likelihood loss.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def softmax(x):
        numer = np.exp(x - x.max(axis=1, keepdims=True))
        denom = numer.sum(axis=1, keepdims=True)
        return numer / denom

    def forward(self, inputs, y_target):
        assert len(inputs) == len(y_target)
        n_obs = len(y_target)
        probs = self.softmax(inputs)
        self._cache_current = y_target, probs

        out = -1 / n_obs * np.sum(y_target * np.log(probs))
        return out

    def backward(self):
        y_target, probs = self._cache_current
        n_obs = len(y_target)
        return -1 / n_obs * (y_target - probs)


class SigmoidLayer(Layer):
    """
    SigmoidLayer: Applies sigmoid function elementwise.
    """

    def __init__(self):
        self._cache_current = None

    def forward(self, x):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        self._cache_current = x
        return 1 / (1 + np.exp(-x))

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        sig = self.forward(self._cache_current)
        return grad_z * sig * (1 - sig)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class ReluLayer(Layer):
    """
    ReluLayer: Applies Relu function elementwise.
    """

    def __init__(self):
        self._cache_current = None

    def forward(self, x):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        self._cache_current = x
        return np.maximum(0, x)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        grad_z[self._cache_current <= 0] = 0
        return grad_z

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class LinearLayer(Layer):
    """
    LinearLayer: Performs affine transformation of input.
    """

    def __init__(self, n_in, n_out):
        """Constructor.

        Arguments:
            n_in {int} -- Number (or dimension) of inputs.
            n_out {int} -- Number (or dimension) of outputs.
        """
        self.n_in = n_in
        self.n_out = n_out

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        self._W = xavier_init((self.n_in, self.n_out))
        self._b = np.zeros((1, self.n_out))

        # dims (batch_size, n_in)
        self._cache_current = None

        # dims (n_in, n_out)
        self._grad_W_current = None

        # dims (1, n_out)
        self._grad_b_current = None

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, x):
        """
        Performs forward pass through the layer (i.e. returns Wx + b).

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._cache_current = x
        return np.dot(x, self._W) + self._b

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        m = grad_z.shape[0]
        self._grad_W_current = np.dot(self._cache_current.T, grad_z) / m
        self._grad_b_current = np.sum(grad_z, axis=0, keepdims=True) / m

        return np.dot(grad_z, self._W.T)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        layer's parameters using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._W -= self._grad_W_current * learning_rate
        self._b -= self._grad_b_current * learning_rate

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class MultiLayerNetwork(object):
    """
    MultiLayerNetwork: A network consisting of stacked linear layers and
    activation functions.
    """

    def __init__(self, input_dim, neurons, activations):
        """Constructor.

        Arguments:
            input_dim {int} -- Dimension of input (excluding batch dimension).
            neurons {list} -- Number of neurons in each layer represented as a
                list (the length of the list determines the number of layers).
            activations {list} -- List of the activation function to use for
                each layer.
        """
        self.input_dim = input_dim
        self.neurons = neurons
        self.activations = activations

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # raise errors for incorrect arguments
        if len(neurons) != len(activations):
            raise Exception(
                "Number of neurons must match number of activations")

        layers = [LinearLayer(input_dim, neurons[0])]

        for i, activation in enumerate(activations):

            if activation is "relu":
                activation_layer = ReluLayer()
            elif activation is "sigmoid":
                activation_layer = SigmoidLayer()
            elif activation is "identity":
                activation_layer = None
            else:
                raise Exception('Non-supported activation function')

            if activation_layer:
                layers.append(activation_layer)

            if i+1 < len(activations):
                layers.append(LinearLayer(neurons[i], neurons[i+1]))

        self._layers = np.array(layers)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, x):
        """
        Performs forward pass through the network.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size,
                # _neurons_in_final_layer)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        output = x
        for layer in self._layers:
            output = layer(output)

        return output

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_z):
        """
        Performs backward pass through the network.

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (1,
                # _neurons_in_final_layer).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, input_dim).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        grad = grad_z
        for layer in np.flip(self._layers):
            grad = layer.backward(grad)

        return grad

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        parameters of all layers using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        for layer in self._layers:
            layer.update_params(learning_rate)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_network(network, fpath):
    """
    Utility function to pickle `network` at file path `fpath`.
    """
    with open(fpath, "wb") as f:
        pickle.dump(network, f)


def load_network(fpath):
    """
    Utility function to load network found at file path `fpath`.
    """
    with open(fpath, "rb") as f:
        network = pickle.load(f)
    return network


class Trainer(object):
    """
    Trainer: Object that manages the training of a neural network.
    """

    def __init__(
        self,
        network,
        batch_size,
        nb_epoch,
        learning_rate,
        loss_fun,
        shuffle_flag,
    ):
        """Constructor.

        Arguments:
            network {MultiLayerNetwork} -- MultiLayerNetwork to be trained.
            batch_size {int} -- Training batch size.
            nb_epoch {int} -- Number of training epochs.
            learning_rate {float} -- SGD learning rate to be used in training.
            loss_fun {str} -- Loss function to be used. Possible values: mse,
                bce.
            shuffle_flag {bool} -- If True, training data is shuffled before
                training.
        """
        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._loss_layer = None
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    @staticmethod
    def shuffle(input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, ).

        Returns: 2-tuple of np.ndarray: (shuffled inputs, shuffled_targets).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        indices = np.arange(input_dataset.shape[0])
        np.random.shuffle(indices)
        input_dataset = input_dataset[indices]
        target_dataset = target_dataset[indices]
        return input_dataset, target_dataset

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def train(self, input_dataset, target_dataset):
        """
        Main training loop. Performs the following steps `nb_epoch` times:
            - Shuffles the input data (if `shuffle` is True)
            - Splits the dataset into batches of size `batch_size`.
            - For each batch:
                - Performs forward pass through the network given the current
                batch of inputs.
                - Computes loss.
                - Performs backward pass to compute gradients of loss with
                respect to parameters of network.
                - Performs one step of gradient descent on the network
                parameters.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_training_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_training_data_points, ).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        for epoch in range(self.nb_epoch):
            # Shuffling
            if (self.shuffle_flag):
                input_dataset, target_dataset = \
                    self.shuffle(input_dataset, target_dataset)
            # Splitting (if it can't be split in evenly sized batches, the last
            # batch has fewer elements)
            if (self.batch_size < 1):
                raise ValueError('Batch size must be greater than 0')

            splits = np.arange(
                self.batch_size, input_dataset.shape[0], self.batch_size)

            input_dataset_split = np.split(input_dataset, splits)
            target_dataset_split = np.split(target_dataset, splits)
            # Iterating through batches
            for index, (input_batch, target_batch) in enumerate(zip(input_dataset_split, target_dataset_split)):
                # Calculate predictions
                predictions = self.network(input_batch)
                # Evaluate loss
                if (self.loss_fun == "mse"):
                    loss_function = MSELossLayer()
                elif (self.loss_fun == "cross_entropy"):
                    loss_function = CrossEntropyLossLayer()
                else:
                    raise Exception('Non-supported loss function')

                loss = loss_function.forward(predictions, target_batch)
                # Backpropagate loss
                self.network.backward(loss_function.backward())
                # Update weights
                self.network.update_params(self.learning_rate)
                # Print loss
                print("[" + str(epoch + 1) + ", " +
                      str(index + 1) + "] loss : " + str(loss))

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def eval_loss(self, input_dataset, target_dataset):
        """
        Function that evaluate the loss function for given data.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_evaluation_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_evaluation_data_points, ).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        if (self.loss_fun not in ["mse", "cross_entropy"]):
            raise ValueError(
                'Loss function unknown. Please use mse or cross_entropy')
        # Calculate predictions
        predictions = self.network.forward(input_dataset)
        # Evaluate loss
        if (self.loss_fun == "mse"):
            loss_function = MSELossLayer()
        elif (self.loss_fun == "cross_entropy"):
            loss_function = CrossEntropyLossLayer()

        return loss_function.forward(predictions, target_dataset)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class Preprocessor(object):
    """
    Preprocessor: Object used to apply "preprocessing" operation to datasets.
    The object can also be used to revert the changes.
    """

    def __init__(self, data):
        """
        Initializes the Preprocessor according to the provided dataset.
        (Does not modify the dataset.)

        Arguments:
            - data {np.ndarray} dataset used to determined the parameters for
            the normalization.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        self.original_interval = np.array([np.amin(data), np.amax(data)])
        self.norm_interval = np.array([0, 1])

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def apply(self, data):
        """
        Apply the pre-processing operations to the provided dataset.

        Arguments:
            - data {np.ndarray} dataset to be normalized.

        Returns:
            {np.ndarray} normalized dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                data[row][col] = self.norm_interval[0] + \
                    (data[row][col] - self.original_interval[0]) * \
                    (self.norm_interval[1] - self.norm_interval[0]) / \
                    (self.original_interval[1] - self.original_interval[0])
        return data

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def revert(self, data):
        """
        Revert the pre-processing operations to retreive the original dataset.

        Arguments:
            - data {np.ndarray} dataset for which to revert normalization.

        Returns:
            {np.ndarray} reverted dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                data[row][col] = self.original_interval[0] + \
                    (data[row][col] - self.norm_interval[0]) * \
                    (self.original_interval[1] - self.original_interval[0]) / \
                    (self.norm_interval[1] - self.norm_interval[0])
        return data

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def example_main():
    '''
    # TESTING PREPROCESSOR

    data = np.loadtxt("iris.dat")
    prep = Preprocessor(data)
    print("Original Dataset")
    print(data)

    # Normalizing
    normalized_dataset = prep.apply(data)
    print("Normalized Dataset")
    print(normalized_dataset)

    # Reverting
    reverted_dataset = prep.revert(normalized_dataset)
    print("Original Dataset")
    print(reverted_dataset)

    # TESTING ACTIVATION FUNCTIONS

    # Sigmoid
    data = np.loadtxt("iris.dat")
    print(data)
    sig = SigmoidLayer()
    processed_data = sig.forward(data)
    print(processed_data)
    backward_data = sig.backward(data)
    print(backward_data)

    # ReLu
    data = np.loadtxt("iris.dat")
    print(data)
    relu = ReluLayer()
    processed_data = relu.forward(data)
    print(processed_data)
    backward_data = relu.backward(data)
    print(backward_data)


    # TESTING TRAINER

    # shuffle
    trainer = Trainer(None, None, None, None, None, None)
    a_i = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    a_t = np.array([1, 2, 3, 4])
    print("Original")
    print(a_i)
    print(a_t)
    print()
    a_i, a_t = trainer.shuffle(a_i, a_t)
    print("Shuffled")
    print(a_i)
    print(a_t)
    print()

    '''

    # ORIGINAL TEST FUNCTION (provided)
    input_dim = 4
    neurons = [16, 3]
    activations = ["relu", "relu"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    dat = np.loadtxt("iris.dat")
    np.random.shuffle(dat)

    x = dat[:, :4]
    y = dat[:, 4:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    trainer = Trainer(
        network=net,
        batch_size=2,
        learning_rate=0.01,
        nb_epoch=2000,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )

    trainer.train(x_train_pre, y_train)

    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print("Validation accuracy: {}".format(accuracy))


if __name__ == "__main__":
    example_main()
