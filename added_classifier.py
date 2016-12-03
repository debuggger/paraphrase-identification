import pickle
import tflearn
import numpy as np

FILE_NAME = "./data/new_naive_data.dump"


def get_data():
    data = pickle.load(open(FILE_NAME, 'r'))
    vectors = np.array([entry[0] for entry in data])
    labels = np.array([(1, 0) if int(entry[1]) == 1 else (0, 1) for entry in data])
    vectors, labels = _shuffle_in_unison(vectors, labels)
    no_of_entries = len(vectors)

    division_point = int(0.9 * no_of_entries)
    training_vectors = vectors[0: division_point, :]
    training_labels = labels[0: division_point, :]
    test_vectors = vectors[division_point + 1:, :]
    test_labels = vectors[division_point + 1:, :]
    return dict(train=training_vectors, train_labels=training_labels, test=test_vectors, test_labels=test_labels)


def _shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def run_classifier(dataset):
    # Create model
    input_layer = tflearn.input_data(shape=[None, 200])
    dense_1 = tflearn.fully_connected(input_layer, 128, activation='relu')
    # dense_2 = tflearn.fully_connected(dense_1, 16, activation='relu')
    softmax = tflearn.fully_connected(dense_1, 2, activation='softmax')

    # Regression
    sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.1, decay_step=10000)
    top_k = tflearn.metrics.Top_k(5)
    net = tflearn.regression(softmax, optimizer=sgd, metric=top_k, loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0, session=None)
    model.fit(dataset["train"], dataset["train_labels"], n_epoch=30, show_metric=True,
              run_id="fold_training")

    # Test model performance
    match_count = 0.0
    predictions = model.predict(dataset["test"])
    for index in xrange(0, len(predictions)):
        if compare(predictions[index], dataset["test_labels"][index]):
            match_count += 1

    print ("Accuracy on Fresh Data is : {}".format(str(match_count / len(predictions))))


def compare(prediction_vector, label_vector):
    if (label_vector[0] > label_vector[1] and prediction_vector[0] > prediction_vector[1]) or \
            (label_vector[1] > label_vector[0] and prediction_vector[1] > prediction_vector[0]):
        return True
    else:
        return False


def main():
    run_classifier(get_data())

if __name__ == "__main__":
    main()
