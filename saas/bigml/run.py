from saas_bigml import bigml
import sys, os

# probably quite unpythonic..
current_path = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.realpath(current_path+"/../../misc"))
from classification_metrics import classification_metrics
from regression_metrics import regression_metrics

all_datasets = {
    'classification': [
        'iris',
        'abalone',
        'cardiotocography',
        'census',
    ], 
    'regression': [
        'airfoil',
        'concrete',
        'housing',   
        'protein-tertiary-structure'
    ], 
}

for ml_type in all_datasets:
    for dataset in all_datasets[ml_type]:

        print('\n%s dataset:\n' % dataset.title() )
        bigml(
            current_path+'/../../data/%s/train_last_field.csv' % dataset,
            current_path+'/../../data/%s/test_without_results.csv' % dataset,
            current_path+'/../../data/%s/bigml_results.csv' % dataset,
        )
        print('\nEvaluation:\n')
        if ml_type == 'classification':
            classification_metrics(
                current_path+'/../../data/%s/test_last_field.csv' % dataset,
                current_path+'/../../data/%s/bigml_results.csv' % dataset,
                'last_field'
            )
        if ml_type == 'regression':
            regression_metrics(
                current_path+'/../../data/%s/test_last_field.csv' % dataset,
                current_path+'/../../data/%s/bigml_results.csv' % dataset,
                'last_field'
            )
