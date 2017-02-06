import numpy as np


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


class RandomClassifier:
    # simple Random Classifier to be used as a borderline
    # comparison of the rest of the approaches used

    # pass in the frequencies of each document class
    # then each prediction is random
    # but the more the occurrences of each class,
    # the higher the probability it will predict it
    def __init__(self, freqs):
        # argument freqs should be a dictionary object
        # with key value pairs of the form:
        # class_name: number_of_occurrences
        self.frequencies = freqs

    def predict(self, the_test_data):
        # returns an ndarray with dtype string_
        # with the same size as the_test_data ndarray
        low = 1
        high = np.sum(self.frequencies.values())
        high = float(high)
        predictions_np = np.random.uniform(low=low, high=high, size=(len(the_test_data), ))

        class_names_list = self.frequencies.keys()

        limits = []
        for class_name in class_names_list:
            limits.append(self.frequencies[class_name])

        for i in range(1, len(limits)):
            limits[i] += limits[i-1]

        predictions = []
        for predicted_value in predictions_np:
            if predicted_value < limits[0]:
                predictions.append(class_names_list[0])
            elif predicted_value < limits[1]:
                predictions.append(class_names_list[1])
            elif predicted_value < limits[2]:
                predictions.append(class_names_list[2])
            elif predicted_value < limits[3]:
                predictions.append(class_names_list[3])
            elif predicted_value < limits[4]:
                predictions.append(class_names_list[4])
            elif predicted_value < limits[5]:
                predictions.append(class_names_list[5])
            elif predicted_value < limits[6]:
                predictions.append(class_names_list[6])
            elif predicted_value < limits[7]:
                predictions.append(class_names_list[7])
            else:
                print("Error of predicted value, shouldn't be such high value: ", predicted_value)
        return predictions


def get_class_frequencies(labels):
    freqs = {}
    for i in range(len(labels)):
        if labels[i] in freqs.keys():
            freqs[labels[i]] += 1
        else:
            freqs[labels[i]] = 1

    return freqs

if __name__ == "__main__":
    train_data, train_labels = load_train_data_and_labels()
    test_data, test_labels = load_test_data_and_labels()
    class_frequencies = get_class_frequencies(train_labels)

    random_classifier = RandomClassifier(class_frequencies)
    predicted_labels = random_classifier.predict(test_data)

    print("Metrics for Random Classifier")
    print("Accuracy = " + str(np.mean(predicted_labels == test_labels)))

    get_metrics(test_labels, predicted_labels)
    print("==============================")
    print("==============================")
    print("")
