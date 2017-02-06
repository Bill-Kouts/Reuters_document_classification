# use of a neural network architecture for text classification
# library used keras, url: keras.io


from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
np.random.seed(1337)  # for reproducibility


def get_data(path):
    # both train and test data files have the same format
    # so we only need one function to parse the data
    # returns a list of tuples, of the form tuple(the_text, the_class_name)

    # the list to store our tuples
    the_data = []

    with open(path) as dataFile:
        for line in dataFile:
            # each line begins with the name of the class and then follows the actual body of text
            # we turn this into a tuple of the form tuple(the_text, the_class_name)
            # and add it to the data list item
            the_data.append(get_single_data_item(line))

    return the_data


def get_single_data_item(line):
    # get the tuple item from each line of data text
    words = line.split()

    # the class of the document is always the first term
    the_class_name = words[0]

    # after the first term, the data follows
    text_data = " ".join(words[1:])

    data_item = (text_data, the_class_name)
    return data_item


def get_train_data(path="r8-train-stemmed.txt"):
    # get the train data from the file r8-train-stemmed.txt
    # return a list of the format: tuple(the_text, the_class_name)
    # the file should be in the same folder with this file
    the_train_data = get_data(path)
    return the_train_data


def get_test_data(path="r8-test-stemmed.txt"):
    # get the test data from the file r8-test-stemmed.txt
    # return a list of the format: tuple(the_text, class_name)
    # the file should be in the same folder with this file
    the_test_data = get_data(path)
    return the_test_data


def load_train_data_and_labels(path="r8-train-stemmed.txt"):
    return load_data_and_labels(path)


def load_test_data_and_labels(path="r8-test-stemmed.txt"):
    return load_data_and_labels(path)


def load_data_and_labels(path):
    data_list_of_tuples = get_data(path)

    # number of data texts
    train_data_size = len(data_list_of_tuples)

    texts = []
    labels = []
    # max size of single data text
    max_size_of_text = 1
    max_size_of_label = 1

    for i in range(train_data_size):
        texts.append(data_list_of_tuples[i][0])
        labels.append(data_list_of_tuples[i][1])

        temp_text_size = len(data_list_of_tuples[i][0])
        temp_label_size = len(data_list_of_tuples[i][1])
        if temp_text_size > max_size_of_text:
            max_size_of_text = temp_text_size
        if temp_label_size > max_size_of_label:
            max_size_of_label = temp_label_size

    my_nd_type_texts = np.dtype((str, max_size_of_text))
    my_nd_type_labels = np.dtype((str, max_size_of_label))
    data_texts_as_ndarray = np.array(texts, dtype=my_nd_type_texts)
    labels_as_ndarray = np.array(labels, dtype=my_nd_type_labels)
    return data_texts_as_ndarray, labels_as_ndarray


def get_metrics(labels_test, labels_predicted):
    print("Metrics calculation started")
    # count TP, FP, TN, FN for micro-averaging and macro-averaging
    tp_index = 0
    fp_index = 1
    tn_index = 2
    fn_index = 3

    # metrics data description: "class_name":[TP_COUNT, FP_COUNT, TN_COUNT, FN_COUNT]

    metrics = {"acq": [0.0, 0.0, 0.0, 0.0], "crude": [0.0, 0.0, 0.0, 0.0], "earn": [0.0, 0.0, 0.0, 0.0],
               "grain": [0.0, 0.0, 0.0, 0.0], "interest": [0.0, 0.0, 0.0, 0.0], "money-fx": [0.0, 0.0, 0.0, 0.0],
               "ship": [0.0, 0.0, 0.0, 0.0], "trade": [0.0, 0.0, 0.0, 0.0]}

    for i in range(len(labels_test)):
        # if the prediction is correct: TP for the correct class and TN for the rest of the classes
        if labels_predicted[i] == labels_test[i]:
            # true positive detected for correct class
            metrics[labels_predicted[i]][tp_index] += 1
            # true negative detected for the rest of the classes
            for other_class in metrics.keys():
                # skip the same class
                if other_class == labels_predicted[i]:
                    continue
                # increase the TN on the rest classes
                metrics[other_class][tn_index] += 1
        # if the prediction is wrong: FP for the predicted class,
        # FN for the correct class and TN for the rest of the classes
        else:
            metrics[labels_predicted[i]][fp_index] += 1
            metrics[labels_test[i]][fn_index] += 1
            for other_class in metrics.keys():
                if other_class == labels_predicted[i] or other_class == labels_test[i]:
                    continue
                metrics[other_class][tn_index] += 1

    print("Metrics calculation ended")

    # for each class calculate precision and recall
    # then for the system calculate micro-averaging and macro-averaging
    print("Precision, Recall, Micro-Averaging and Macro-Averaging calculation started")
    precisions = {"acq": 0, "crude": 0, "earn": 0,
                  "grain": 0, "interest": 0, "money-fx": 0,
                  "ship": 0, "trade": 0}
    recalls = {"acq": 0, "crude": 0, "earn": 0,
               "grain": 0, "interest": 0, "money-fx": 0,
               "ship": 0, "trade": 0}
    micro_averaging_precision = 0
    micro_averaging_recall = 0
    macro_averaging_precision = 0
    macro_averaging_recall = 0

    for key in precisions:
        try:
            precisions[key] = float(metrics[key][tp_index]) / float(metrics[key][tp_index] + metrics[key][fp_index])
        except ZeroDivisionError:
            print("Precision Error calculation for class:", key, "division by zero")

    for key in recalls:
        try:
            recalls[key] = float(metrics[key][tp_index]) / float(metrics[key][tp_index] + metrics[key][fn_index])
        except ZeroDivisionError:
            print("Recall Error calculation in class:", key, "division by zero")

    # calculate sums for all classes for: TPs, TPs+FPs, TPs+FNs
    # which are needed in micro-averaging and macro-averaging
    tp_sum = 0
    tp_fp_sum = 0
    tp_fn_sum = 0
    precisions_sum = 0
    recalls_sum = 0

    for class_name in metrics:
        tp_sum += metrics[class_name][tp_index]
        tp_fp_sum += metrics[class_name][tp_index] + metrics[class_name][fp_index]
        tp_fn_sum += metrics[class_name][tp_index] + metrics[class_name][fn_index]
        precisions_sum += precisions[class_name]
        recalls_sum += recalls[class_name]

    micro_averaging_precision = float(tp_sum) / float(tp_fp_sum)
    micro_averaging_recall = float(tp_sum) / float(tp_fn_sum)

    macro_averaging_precision = float(precisions_sum) / float(len(metrics.keys()))
    macro_averaging_recall = float(recalls_sum) / float(len(metrics.keys()))

    print("Micro and Macro Averaging calculation ended")
    print("")
    print("Micro Averaging Precision:", micro_averaging_precision, "Micro Averaging Recall:", micro_averaging_recall)
    print("Macro Averaging Precision:", macro_averaging_precision, "Macro Averaging Recall:", macro_averaging_recall)

    for class_name in precisions:
        print("Class", class_name, "precision:", precisions[class_name], "recall:", recalls[class_name])


def str_to_int_labels(list_of_strs):
    # class names with corresponding index:
    # acq = 0, crude = 1, earn = 2
    # grain = 3, interest = 4, money-fx = 5
    # ship = 6, trade = 7
    int_list = []
    for each_label in list_of_strs:
        if each_label == 'acq':
            int_list.append(0)
        elif each_label == 'crude':
            int_list.append(1)
        elif each_label == 'earn':
            int_list.append(2)
        elif each_label == 'grain':
            int_list.append(3)
        elif each_label == 'interest':
            int_list.append(4)
        elif each_label == 'money-fx':
            int_list.append(5)
        elif each_label == 'ship':
            int_list.append(6)
        elif each_label == 'trade':
            int_list.append(7)
        else:
            print("ERROR while turning labels into indexes")

    return int_list


def int_to_str_labels(list_of_ints):
    # class names with corresponding index:
    # acq = 0, crude = 1, earn = 2
    # grain = 3, interest = 4, money-fx = 5
    # ship = 6, trade = 7
    str_list = []
    for each_index in list_of_ints:
        if each_index == 0:
            str_list.append('acq')
        elif each_index == 1:
            str_list.append('crude')
        elif each_index == 2:
            str_list.append('earn')
        elif each_index == 3:
            str_list.append('grain')
        elif each_index == 4:
            str_list.append('interest')
        elif each_index == 5:
            str_list.append('money-fx')
        elif each_index == 6:
            str_list.append('ship')
        elif each_index == 7:
            str_list.append('trade')
        else:
            print("ERROR while turning indexes into labels")

    return str_list


if __name__ == "__main__":
    train_data, train_labels = load_train_data_and_labels()
    test_data, test_labels = load_test_data_and_labels()

    # size of batch (number of examples used for each pass through the neural net during training)
    batch_size = 32

    # iterations of training on the train data
    # the whole set of training data will be used 5 times
    # because neural networks need a lot of training
    nb_epoch = 5

    # number of top words to look out for
    max_words = 1000

    print('Vectorizing sequence data...')
    tokenizer = Tokenizer(nb_words=max_words)
    # build the vocabulary
    tokenizer.fit_on_texts(train_data)
    tokenizer.fit_on_texts(test_data)

    # convert the vocabulary to tf idf table
    # options are: binary, count, tfidf, freq
    X_train = tokenizer.texts_to_matrix(train_data, mode='tfidf')
    X_test = tokenizer.texts_to_matrix(test_data, mode='tfidf')
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    # number of different classes
    nb_classes = 8
    # classes should be converted to numbers
    train_labels_indexes = str_to_int_labels(train_labels)
    test_labels_indexes = str_to_int_labels(test_labels)
    print('Convert class vector to binary class matrix (for use with categorical_crossentropy)')
    Y_train = np_utils.to_categorical(train_labels_indexes, nb_classes)
    Y_test = np_utils.to_categorical(test_labels_indexes, nb_classes)
    print('Y_train shape:', Y_train.shape)
    print('Y_test shape:', Y_test.shape)

    print('Building model...')
    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    history = model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1, show_accuracy=True, validation_split=0.1)
    score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1, show_accuracy=True)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    predictions = model.predict_classes(X_test)
    predictions = int_to_str_labels(predictions)
    print("==============================")
    print("==============================")
    print("")
    print("Metrics for Neural Net Classifier")
    print("Accuracy = " + str(np.mean(predictions == test_labels)))

    get_metrics(test_labels, predictions)
    print("==============================")
    print("==============================")
    print("")
