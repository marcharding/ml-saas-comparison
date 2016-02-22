from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import sys
import csv

def regression_metrics( csv_test, csv_result, last_or_first ):

    real_results = []
    predicted_results = []

    with open(csv_test, 'rb') as csv_test_file:
        csv_test_reader = csv.reader(csv_test_file, delimiter=',', quotechar='"')
        for row in csv_test_reader:
            if last_or_first == 'first_field':
                real_results.append(float(row.pop(0)))
            else:
                real_results.append(float(row.pop()))
                
    with open(csv_result, 'rb') as csv_result_file:
        csv_result_reader = csv.reader(csv_result_file, delimiter=',', quotechar='"')
        for row in csv_result_reader:
            if last_or_first == 'first_field':
                predicted_results.append(float(row.pop(0)))
            else:
                predicted_results.append(float(row.pop()))            

    labels = list(set(real_results))           

    print('Explained variance score: %f' % explained_variance_score(real_results, predicted_results))  
    print('Mean squared error: %f' % mean_squared_error(real_results, predicted_results))  
    print('Mean absolute error: %f' % mean_absolute_error(real_results, predicted_results))  
