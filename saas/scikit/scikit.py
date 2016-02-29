from timeit import default_timer as timer
from sklearn import cross_validation
from sklearn import metrics
from sklearn import preprocessing
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import csv
import numpy as np

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError, TypeError:
    return False

def convert_categorical_to_int( data ):
    for counter, option in enumerate(data[0]):
        le = preprocessing.LabelEncoder()
        values = zip(*data)[counter]
        if all(isfloat(item)==True for item in values) == True:
            continue
        le.fit(values)
        new = le.transform(values)
        for subcounter, labels in enumerate(new):
            data[subcounter][counter] = labels
    return data

def scikit( train_csv, test_csv, result_csv, force_model_type = None):

    X = []
    y = []

    with open(train_csv, 'rb') as csv_train_file:
        csv_train_reader = csv.reader(csv_train_file, delimiter=',', quotechar='"')
        for row in csv_train_reader:
            y.append(row.pop(0))
            X.append(row)

    # train model
    start_training = timer()

    X = convert_categorical_to_int( X )
    X = np.array(X, np.float)

    # normalize the data attributes
    # X = preprocessing.normalize(X)
    # standardize the data attributes
    # X = preprocessing.scale(X)

    if force_model_type == 'CLASSIFICATION':
        model = RandomForestClassifier(n_estimators=10)
    elif force_model_type == 'REGRESSION':
        model = LogisticRegression()
    elif force_model_type == 'SVC':
        model = svm.SVC(kernel='linear',C=1)
    else:
        model = tree.DecisionTreeClassifier()

    model.fit(X, y)

    end_training = timer()
    print('Training model.')
    print('Training took %i Seconds.' % (end_training - start_training) );

    X = []
    y = []

    open(result_csv, 'w').close()

    with open(test_csv) as csv_test_file:
        test_csv_reader = csv.reader(csv_test_file, delimiter=',', quotechar='"')
        for row in test_csv_reader:
            y.append(row.pop(0))
            X.append(row)

    # test create_model
    start_test = timer()

    XX = convert_categorical_to_int( X )
    XX = np.array(XX, np.float)

    predicted = model.predict(XX)
    results = [[predicted[ix]] + X[ix] for ix in range(len(predicted))]

    with open(result_csv, 'wb') as csv_result_file:
        result_csv_writer = csv.writer(csv_result_file, delimiter=',', quotechar='"', lineterminator="\n")
        for row in results:
            result_csv_writer.writerow(row)

    end_test = timer()
    print('Testing took %i Seconds' % (end_test - start_test) );
