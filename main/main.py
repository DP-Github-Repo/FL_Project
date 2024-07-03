import numpy as np
import tensorflow as tf
import matplotlib.pyplot as pyplot
from tqdm import trange


def generate_dataset():
    LOWER_END = -10
    UPPER_END = 10
    NUMBER_OF_SAMPLES = 10000

    _X = np.random.uniform(LOWER_END, UPPER_END, NUMBER_OF_SAMPLES // 50)
    UPPER_END_Y = 25
    LOWER_END_Y = -25
    _y = np.array([i/UPPER_END_Y for i in range(LOWER_END_Y, UPPER_END_Y)])

    data = []

    for x_sample in _X:
        for y_sample in _y:
            data.append([x_sample, y_sample])

    data = np.array(data)
    Y_INDEX = 1
    X_INDEX = 0
    labels = np.array([0.0 if i[Y_INDEX] > np.sin(i[X_INDEX]) else
                       1.0 for i in data])
    return data, labels


def generate_model(_X, _y):
    _model = tf.keras.models.Sequential([
        tf.keras.layers.Input((2,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    _model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
    NUMBER_OF_TRAINING_LOOPS = 100
    _history = _model.fit(_X, _y, epochs=NUMBER_OF_TRAINING_LOOPS, validation_split=0.1, verbose=0)
    return _model, _history


def display_results(_history):
    pyplot.plot(_history.history["val_loss"], label="Validation Loss")
    pyplot.plot(history.history["val_acc"], label="Validation Accuracy")
    pyplot.legend()
    pyplot.show()
    return


def client_side_model(initial_weights=None):
    _client_model = tf.keras.models.Sequential([
        tf.keras.layers.Input((2,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    _client_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

    if initial_weights is not None:
        _client_model.set_weights(initial_weights)
    return _client_model


def build_aggregator_model():
    _aggregator_model = client_side_model()

    _weights = _aggregator_model.get_weights()
    _weights = [0*i for i in _weights]

    _aggregator_model.set_weights(_weights)
    return _aggregator_model


def generate_clients(_number_of_clients, _X, _y):
    _clients = {i: [] for i in range(_number_of_clients)}
    indexes = np.arange(_X.shape[0])
    number_of_samples = 10000
    number_of_samples_per_client = number_of_samples // _number_of_clients

    for _client in _clients.keys():
        _clients[_client] = np.random.choice(indexes,
                                             number_of_samples_per_client,
                                             replace=False)
        indexes = np.array([i for i in indexes if i not in _clients[_client]])

    _clients_dataset = {}

    for _client in _clients.keys():
        _clients_dataset.update({_client: [_X[_clients[_client]], _y[_clients[_client]]]})
    return _clients_dataset


if __name__ == "__main__":
    X, y = generate_dataset()
    # print(generate_dataset())

    model, history = generate_model(X, y)
    # print(model.evaluate(X, y))
    # display_results(history)

    # client_model = client_side_model()
    # print(client_model)
    aggregator_model = build_aggregator_model()
    # print(aggregator_model)
    number_of_clients = 10
    clients = generate_clients(number_of_clients, X, y)
    # print(clients)

    aggregator_epochs = 20
    aggregator_weights = aggregator_model.get_weights()
    client_weights = {client: None for client in range(number_of_clients)}
    client_epochs = 10

    aggregator_loss = []
    aggregator_accuracy = []

    for epoch in trange(aggregator_epochs, desc="Aggregator Model"):
        clients_dataset = generate_clients(number_of_clients, X, y)
        for client in client_weights.keys():
            weight = client_weights[client]
            client_model = client_side_model(weight)

            X_INDEX = 0
            Y_INDEX = 1

            client_model.fit(clients_dataset[client][X_INDEX],
                             clients_dataset[client][Y_INDEX],
                             epochs=client_epochs,
                             verbose=False,
                             validation_split=0.1)

            client_weights[client] = client_model.get_weights()

        aggregator_weights = [np.mean([client_weights[client][index] for client
                                      in range(number_of_clients)],
                                      axis=0) for index in range(len(aggregator_weights))]

        client_weights = {client: aggregator_weights for client in client_weights.keys()}
        aggregator_model.set_weights(aggregator_weights)
        loss, accuracy = aggregator_model.evaluate(X, y, verbose=False)

        aggregator_loss.append(loss)
        aggregator_accuracy.append(accuracy)

        pyplot.plot(range(len(aggregator_accuracy)), aggregator_accuracy, label="Accuracy")
        pyplot.plot(range(len(aggregator_loss)), aggregator_loss, label="loss")
        pyplot.legend()
        pyplot.show()
