# this is the base algorithm for a Naive Bayes Classifier
# it will be the main example to be used as a comparison
# against the rest of the algorithms used for this problem

# package used: textblob, url: http://textblob.readthedocs.org/

from textblob.classifiers import NaiveBayesClassifier
import time


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


if __name__ == "__main__":
    train_data = get_train_data()
    test_data = get_test_data()

    # create the simplest classifier
    print("Naive Bayes training started")
    t0 = time.clock()

    nb_classifier = NaiveBayesClassifier(train_data)
    print("Naive Bayes training examples ended: ", str(len(train_data)), "time it took: ", str(time.clock()-t0))

    # use size_of_subset to train and test on a subset of train data and test data
    # because textblob is to slow (nltk naive bayes is behind)
    # size_of_subset = 100
    # nb_classifier = NaiveBayesClassifier(train_data[:size_of_subset])
    # print("Naive Bayes training examples ended: ", str(size_of_subset), "time it took: ", str(time.clock()-t0))

    print("Naive Bayes training ended")

    print("Naive Bayes testing started")
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

    # if size_of_subset used uncomment the following:
    # for entry in test_data[:size_of_subset]
    # and comment the following:
    for entry in test_data:
        predicted_class = nb_classifier.classify(entry[0])
        correct_class = entry[1]

        # if the prediction is correct: TP for the correct class and TN for the rest of the classes
        if predicted_class == correct_class:
            # true positive detected for correct class
            metrics[predicted_class][tp_index] += 1
            # true negative detected for the rest of the classes
            for other_class in metrics.keys():
                # skip the same class
                if other_class == predicted_class:
                    continue
                # increase the TN on the rest classes
                metrics[other_class][tn_index] += 1
        # if the prediction is wrong: FP for the predicted class,
        # FN for the correct class and TN for the rest of the classes
        else:
            metrics[predicted_class][fp_index] += 1
            metrics[correct_class][fn_index] += 1
            for other_class in metrics.keys():
                if other_class == predicted_class or other_class == correct_class:
                    continue
                metrics[other_class][tn_index] += 1

    print("Naive Bayes testing ended")
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
