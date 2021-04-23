import numpy as np
import scipy

from scipy.spatial import distance
import sys

RESULT_KNN = []
K_KNN = 7
RESULT_PERCEPTRON = []
ETA_PERCEPTRON = 0.001
RESULT_PASSIVE_AGGRESIVE = []
EPOCH_PA = 1
EPOCH_PER = 7
CLUSTER = 3
CROSS_VALID = 50


def load_text(text):
    # text = np.genfromtxt(text, dtype="f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, |U2", delimiter=',')
    text = np.genfromtxt(text, dtype=(
        np.float, np.float, np.float, np.float, np.float, np.float, np.float, np.float, np.float, np.float, np.float,
        'U1'), delimiter=',')
    float_list = []
    for line in text:
        if line[11] == 'W':
            line[11] = np.float(0)
        else:
            line[11] = np.float(1)  # for R
        float_list.append(list(map(float, line)))
    text = np.array(float_list)
    return text


def Knn_train(train_x, train_y):
    dict_train_test = []
    for i in range(len(train_x)):
        dict_train_test.append([train_y[i], train_x[i]])
    return dict_train_test


def Knn_test(test_x, dict_train):
    RESULT_KNN.clear()
    distance = []
    for t in test_x:
        for train in dict_train:
            d = scipy.spatial.distance.euclidean(t.real, (train[1]).real)
            distance.append(d)
        np_array_dist = np.array(list(distance), dtype=object)
        sort_array = np.argsort(np_array_dist)[:K_KNN]
        #idx = np.argpartition(np_array_dist, K_KNN)
        find_label_knn(sort_array, [row[0] for row in dict_train])
        distance.clear()


def find_label_knn(idx, type_label):
    counters = [0, 0, 0]
    for i in idx:
        if type_label[i] == 0:
            counters[0] += 1
        elif type_label[i] == 1:
            counters[1] += 1
        else:
            counters[2] += 1
    RESULT_KNN.append(counters.index(max(counters)))


def KNN(train_x, train_y, test_x):
    dict_train = Knn_train(train_x, train_y)
    Knn_test(test_x, dict_train)


def choose_train_and_validition(train_x, train_y):
    length_train_x = len(train_x)
    num = 0
    list_result = []
    ## valid_x, valid_y, train_x, train_y
    while num < length_train_x:
        end = num + CROSS_VALID
        if num + CROSS_VALID > length_train_x:
            end = length_train_x
        row = [train_x[num: end], train_y[num: end]]
        copy_train_x = np.array(train_x, copy=True)
        copy_train_x = np.delete(copy_train_x, slice(num, end), axis=0)
        row.append(copy_train_x)
        copy_train_y = np.array(train_y, copy=True)
        copy_train_y = np.delete(copy_train_y, slice(num, end), axis=0)
        row.append(copy_train_y)
        list_result.append(row)
        num += CROSS_VALID

    return list_result


def normalize(train_x, test_x):
    # normalize train
    max_in_columns = np.amax(train_x, axis=0)
    min_in_columns = np.amin(train_x, axis=0)
    train_x = normalize_according_min_max(train_x, max_in_columns, min_in_columns)
    # normalize test
    test_x = normalize_according_min_max(test_x, max_in_columns, min_in_columns)
    return train_x, test_x


def normalize_according_min_max(list_to_normalizem, max_values, min_values):
    for line in list_to_normalizem:
        for c in range(len(line)):
            to_div = max_values[c] - min_values[c]
            if to_div == 0:
                line[c] = 1
            else:
                line[c] = (line[c] - min_values[c]) / to_div
    return list_to_normalizem


def normalize_all(list_result):
    for l in list_result:
        l[2], l[0] = normalize(l[2], l[0])
    return list_result


def compare_knn(list_result):
    all_knn = []
    count = []
    for i in range(1, 11):
        global K_KNN
        K_KNN = i
        count.clear()
        for l in list_result:
            RESULT_KNN.clear()
            KNN(l[2], l[3], l[0])
            sum_equel = 0
            for i in range(len(RESULT_KNN)):
                if RESULT_KNN[i] == l[1][i]:
                    sum_equel += 1
            count.append(sum_equel * 100 / len(RESULT_KNN))
        all_knn.append(sum(count) / len(count))
    print(all_knn)


def compare_passive_aggresive(list_result):
    all_knn = []
    count = []
    global EPOCH_PA
    EPOCH_PA = 8
    while EPOCH_PA < 20:
        count.clear()
        for l in list_result:
            RESULT_PASSIVE_AGGRESIVE.clear()
            passive_aggresive(l[2], l[3], l[0])
            sum_equel = 0
            sum_result = len(RESULT_PASSIVE_AGGRESIVE)
            for i in range(sum_result):
                if RESULT_PASSIVE_AGGRESIVE[i] == l[1][i]:
                    sum_equel += 1
            count.append(sum_equel * 100 / sum_result)
        all_knn.append([EPOCH_PA, sum(count) / len(count)])
        EPOCH_PA += 1
    print(all_knn)


def compare_perceptron(list_result):
    all_knn = []
    count = []
    #global ETA_PERCEPTRON
    #ETA_PERCEPTRON = 0.05
    global EPOCH_PER
    #while ETA_PERCEPTRON < 1:
    EPOCH_PER = 1
    while EPOCH_PER < 20:
        count.clear()
        for l in list_result:
            RESULT_PERCEPTRON.clear()
            perceptron(l[2], l[3], l[0])
            sum_equel = 0
            sum_result = len(RESULT_PERCEPTRON)
            for i in range(sum_result):
                if RESULT_PERCEPTRON[i] == l[1][i]:
                    sum_equel += 1
            count.append(sum_equel * 100 / sum_result)
        all_knn.append([(ETA_PERCEPTRON, EPOCH_PER), sum(count) / len(count)])
        EPOCH_PER += 1
    #ETA_PERCEPTRON += 0.05
    print(all_knn)


def get_list_to_check(train_x, train_y):
    list_result = choose_train_and_validition(train_x, train_y)
    list_result = normalize_all(list_result)
    return list_result


def remove_filter(train_x, test_x, i):
    copy_train_x = np.delete(train_x, i, axis=1)
    copy_test_x = np.delete(test_x, i, axis=1)
    return copy_train_x, copy_test_x


def add_bias(np_array):
    num_row_array = len(np_array)
    column_one_array = np.ones((num_row_array, 1))
    array_with_bias = np.hstack((column_one_array, np_array))
    return array_with_bias


def my_shuffle(train_x, train_y):
    indices = np.arange(train_x.shape[0])
    np.random.shuffle(indices)
    X_train = train_x[indices]
    Y_train = train_y[indices]
    return X_train, Y_train


def get_predict(w, x):
    return np.argmax(np.dot(w, x))


def train_perceptron(train_x, train_y):
    w = np.zeros((CLUSTER, len(train_x[0])))
    for e in range(EPOCH_PER):
        X_train, Y_train = my_shuffle(train_x, train_y)
        for x, y in zip(X_train, Y_train):
            y_hat = get_predict(w, x)
            y = int(y)
            y_hat = int(y_hat)
            if y is not y_hat:
                w[y, :] = w[y, :] + ETA_PERCEPTRON * x
                w[y_hat, :] = w[y_hat, :] - ETA_PERCEPTRON * x
    return w


def train_passive_aggresive(train_x, train_y):
    w = np.zeros((CLUSTER, len(train_x[0])))
    X_train, Y_train = my_shuffle(train_x, train_y)
    for x, y in zip(X_train, Y_train):
        y = int(y)
        mult_array = np.dot(w, x)
        mult_array[y] = np.NINF
        y_hat = np.argmax(mult_array)
        y_hat = int(y_hat)
        tau = calculate_tau(w, x, y, y_hat)
        w[y, :] = w[y, :] + tau * x
        w[y_hat, :] = w[y_hat, :] - tau * x
    return w


def calculate_tau(w, x, y, y_hat):
    loss = max(0, (1 - np.dot(w[y, :], x) + np.dot(w[y_hat, :], x)))
    norm_x = np.power(np.linalg.norm(x), 2) * 2
    if norm_x == 0:
        return 0
    return loss / norm_x


def passive_aggresive(train_x, train_y, test_x):
    train_x = add_bias(train_x)
    test_x = add_bias(test_x)
    w = train_passive_aggresive(train_x, train_y)
    test_passive_aggresive(w, test_x)


def test_passive_aggresive(w, test_x):
    RESULT_PASSIVE_AGGRESIVE.clear()
    for line in test_x:
        y_test = get_predict(w, line)
        RESULT_PASSIVE_AGGRESIVE.append(y_test)


def test_perceptron(w, test_x):
    RESULT_PERCEPTRON.clear()
    for line in test_x:
        y_test = get_predict(w, line)
        RESULT_PERCEPTRON.append(y_test)


def perceptron(train_x, train_y, test_x):
    train_x = add_bias(train_x)
    test_x = add_bias(test_x)
    w = train_perceptron(train_x, train_y)
    test_perceptron(w, test_x)


def main():
    args = sys.argv
    train_x = args[1]
    train_y = args[2]
    test_x = args[3]

    train_y = np.loadtxt(train_y)
    train_x = load_text(train_x)
    test_x = load_text(test_x)
    train_x, test_x = normalize(train_x, test_x)
    KNN(train_x, train_y, test_x)
    perceptron(train_x, train_y, test_x)
    passive_aggresive(train_x, train_y, test_x)
    for t in range(len(RESULT_KNN)):
        print(f"knn: {RESULT_KNN[t]}, perceptron: {RESULT_PERCEPTRON[t]}, pa: {RESULT_PASSIVE_AGGRESIVE[t]}")



if __name__ == "__main__":
    main()
