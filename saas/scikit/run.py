from scikit import scikit
import sys, os

# probably quite unpythonic..
current_path = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.realpath(current_path+"/../../misc"))
from classification_metrics import classification_metrics
from regression_metrics import regression_metrics

all_datasets = {
    'CLASSIFICATION': [
        'iris',
        'abalone',
        'cardiotocography',
        'census',
    ],
    'REGRESSION': [
        'airfoil',
        'concrete',
        'housing',
        # 'protein-tertiary-structure'
    ],
}

for ml_type in all_datasets:
    for dataset in all_datasets[ml_type]:

        print('\n%s dataset:\n' % dataset.title() )
        scikit(
            current_path+'/../../data/%s/train_first_field.csv' % dataset,
            current_path+'/../../data/%s/test_first_field.csv' % dataset,
            current_path+'/../../data/%s/scikit_results.csv' % dataset,
            force_model_type=ml_type
        )
        print('\nEvaluation:\n')
        if ml_type == 'CLASSIFICATION':
            classification_metrics(
                current_path+'/../../data/%s/test_first_field.csv' % dataset,
                current_path+'/../../data/%s/scikit_results.csv' % dataset,
                'first_field'
            )
        if ml_type == 'REGRESSION':
            regression_metrics(
                current_path+'/../../data/%s/test_first_field.csv' % dataset,
                current_path+'/../../data/%s/scikit_results.csv' % dataset,
                'first_field'
            )
