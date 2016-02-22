from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
import sys
import csv

def classification_metrics( csv_test, csv_result, last_or_first ):

    real_results = []
    predicted_results = []

    with open(csv_test, 'rb') as csv_test_file:
        csv_test_reader = csv.reader(csv_test_file, delimiter=',', quotechar='"')
        for row in csv_test_reader:
            if last_or_first == 'first_field':
                real_results.append(row.pop(0))
            else:
                real_results.append(row.pop())
                
    with open(csv_result, 'rb') as csv_result_file:
        csv_result_reader = csv.reader(csv_result_file, delimiter=',', quotechar='"')
        for row in csv_result_reader:
            if last_or_first == 'first_field':
                predicted_results.append(row.pop(0))
            else:
                predicted_results.append(row.pop())            

    labels = list(set(real_results))

    # print predicted_results       
    print("Recall Score: %f" % recall_score(real_results, predicted_results,pos_label=labels[0],labels=labels,average='micro'))    
    print("Precision Score: %f" % precision_score(real_results, predicted_results,pos_label=labels[0],labels=labels,average='micro'))   
    print("F1 Score: %f" % f1_score(real_results, predicted_results,pos_label=labels[0],labels=labels,average='micro'))  
    print('\n')
    print(classification_report(real_results, predicted_results))  
