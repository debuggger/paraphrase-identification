import pickle
import tflearn
import numpy as np

FILE_NAME = "./data/new_naive_data.dump"


def get_data():
    data = pickle.load(open(FILE_NAME, 'r'))
    vectors = np.array([entry[0] for entry in data])
    labels = np.array([(0, 1) if int(entry[1]) == 1 else (1, 0) for entry in data])
    vectors, labels = _shuffle_in_unison(vectors, labels)
    no_of_entries = len(vectors)

    division_point = int(0.9 * no_of_entries)
    training_vectors = vectors[0: division_point, :]
    training_labels = labels[0: division_point, :]
    test_vectors = vectors[division_point + 1:, :]
    test_labels = labels[division_point + 1:, :]
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
    dense_2 = tflearn.fully_connected(dense_1, 16, activation='linear')
    # dense_3 = tflearn.fully_connected(dense_2, 4, activation='relu')
    softmax = tflearn.fully_connected(dense_2, 2, activation='softmax')

    # Regression
    sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.9, decay_step=100)
    top_k = tflearn.metrics.Top_k(2)
    net = tflearn.regression(softmax, optimizer=sgd, metric=top_k, loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0, session=None)
    model.fit(dataset["train"], dataset["train_labels"], n_epoch=30, show_metric=True,
              run_id="fold_training")

    # Test model performance
    match_count = 0.0
    zero_count = 0.0
    zero_match = 0.0
    one_count = 0.0
    one_match = 0.0
    
    predictions = model.predict(dataset["test"])
    
    for index in xrange(0, len(predictions)):
        is_zero = True if np.all(np.equal(dataset["test_labels"][index], np.array((0, 1)))) else False
        if is_zero:
            zero_count += 1
        else:
            one_count += 1
        if compare(predictions[index], dataset["test_labels"][index]):
            match_count += 1
            if is_zero:
                zero_match += 1
            else:
                one_match += 1

    print ("Accuracy on Fresh Data is : {} %".format(str(100.0 * match_count / len(predictions))))
    print ("Accuracy on Zeros is : {} %".format(str(100 * zero_match / zero_count)))
    print ("Accuracy on Ones is : {} %".format(str(100 * one_match / one_count)))

    # print ("Total Zeros (0,1) : {} \tAccuracy : {} %".format(str(zero_count), str(100.0 * zero_match / zero_count)))
    # print ("Total Ones (1,0) : {} \tAccuracy : {} %".format(str(one_count), str(100.00 * one_match / one_count)))


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
