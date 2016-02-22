from bigml.api import BigML
import csv
import time
import sys
from timeit import default_timer as timer

def bigml( train_csv, test_csv, result_csv ):

    api = BigML(dev_mode=True)

    # train model
    start_training = timer()

    source_train = api.create_source(train_csv)
    dataset_train = api.create_dataset(source_train)
    model = api.create_model(dataset_train)

    end_training = timer()
    print('Training model.')
    print('Training took %i Seconds.' % (end_training - start_training) ); 

    # test create_model
    start_test = timer()

    source_test = api.create_source(test_csv)
    dataset_test = api.create_dataset(source_test)

    batch_prediction = api.create_batch_prediction(
        model, 
        dataset_test,
        {
            "name": "census prediction", 
            "all_fields": True,
            "header": False,
            "confidence": False
        }
    )

    # wait until batch processing is finished
    while api.get_batch_prediction(batch_prediction)['object']['status']['progress'] != 1:
        print api.get_batch_prediction(batch_prediction)['object']['status']['progress']
        time.sleep(1)

    end_test = timer()
    print('Testing took %i Seconds' % (end_test - start_test) ); 

    api.download_batch_prediction(batch_prediction['resource'], filename=result_csv)

    # cleanup
    api.delete_source(source_train)
    api.delete_source(source_test)
    api.delete_dataset(dataset_train)
    api.delete_dataset(dataset_test)
    api.delete_model(model)
