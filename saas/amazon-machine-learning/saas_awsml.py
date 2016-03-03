from timeit import default_timer as timer
import boto3
import random
import awspyml
import time
import json
import csv
import gzip
import os
import copy

def awsml( train_csv, test_csv, result_csv, force_model_type = None):

    s3 = boto3.resource('s3')
    ml = boto3.client('machinelearning')

    with open(train_csv) as csv_train_file:
        test_train_reader = csv.reader(csv_train_file, delimiter=',', quotechar='"')
        row = next(test_train_reader)
        target_variable_number = len(row)

    # create random id fo this set
    random_id = "%i-%i" % (random.randint(0, 65535), random.randint(0, 65535))
    # bucket
    ml_bucket_id = "ml-bucket-id-%s" % random_id
    s3.create_bucket(Bucket=ml_bucket_id)

    data = open(train_csv, 'rb')
    s3.Bucket(ml_bucket_id).put_object(Key='train.csv', Body=data)

    data = open(test_csv, 'rb')
    s3.Bucket(ml_bucket_id).put_object(Key='test.csv', Body=data)

    bucket_policy_dict = {
      "Version": "2012-10-17",
      "Statement": [
          {
              "Sid": "AmazonML_s3:ListBucket",
              "Effect": "Allow",
              "Principal": {
                  "Service": "machinelearning.amazonaws.com"
              },
              "Action": "s3:ListBucket",
              "Resource": "arn:aws:s3:::%s" % ml_bucket_id
          },
          {
              "Sid": "AmazonML_s3:GetObject",
              "Effect": "Allow",
              "Principal": {
                  "Service": "machinelearning.amazonaws.com"
              },
              "Action": "s3:GetObject",
              "Resource": "arn:aws:s3:::%s/*" % ml_bucket_id
          },
          {
              "Sid": "AmazonML_s3:PutObject",
              "Effect": "Allow",
              "Principal": {
                  "Service": "machinelearning.amazonaws.com"
              },
              "Action": "s3:PutObject",
              "Resource": "arn:aws:s3:::%s/*" % ml_bucket_id
          }
      ]
    }
    bucket_policy = s3.BucketPolicy("ml-bucket-id-%s" % random_id).put(Policy=json.dumps(bucket_policy_dict))

    schema = awspyml.SchemaGuesser().from_file(train_csv, header_line=False, target_variable=None)

    if force_model_type == 'classification':
        schema.set_variable_type(target_variable_number-1,"CATEGORICAL")
    if force_model_type == 'regression':
        schema.set_variable_type(target_variable_number-1,"NUMERIC")

    schema_train = copy.deepcopy(schema)
    schema_train.set_target('Var%02d' % target_variable_number)
    # print schema_train.as_json_string()

    schema_test = copy.deepcopy(schema)

    response = ml.create_data_source_from_s3(
        DataSourceId='ml-datasource-id-%s' % random_id,
        DataSourceName='ml-datasource-id-%s' % random_id,
        DataSpec={
            'DataLocationS3': 's3://%s/train.csv' % ml_bucket_id,
            'DataSchema': schema_train.as_json_string()
        },
        ComputeStatistics=True
    )

    schema_test.delete_variable_by_idx(target_variable_number-1)
    # print schema_test.as_json_string()

    response = ml.create_data_source_from_s3(
        DataSourceId='ml-datasource-id-test-%s' % random_id,
        DataSourceName='ml-datasource-id-test%s' % random_id,
        DataSpec={
            'DataLocationS3': 's3://%s/test.csv' % ml_bucket_id,
            'DataSchema': schema_test.as_json_string()
        },
        ComputeStatistics=True
    )

    # print response

    # train model
    start_training = timer()

    if force_model_type == 'classification':
        model_type = 'MULTICLASS'
    elif force_model_type == 'regression':
        model_type = 'REGRESSION'
    else:
        model_type = 'BINARY'

    response = ml.create_ml_model(
        MLModelId='ml-model-id-%s' % random_id,
        MLModelName='ml-model-id-%s' % random_id,
        MLModelType=model_type,
        TrainingDataSourceId='ml-datasource-id-%s' % random_id
    )

    # print response

    while ml.get_ml_model(MLModelId='ml-model-id-%s' % random_id, Verbose=True)['Status'] != 'COMPLETED':
        # print "DEBUG: Training model"
        time.sleep(8)

    end_training = timer()
    print('Training took %i Seconds.' % (end_training - start_training) );

    # test model
    start_test = timer()

    # print "DEBUG: Testing model"
    response = ml.create_batch_prediction(
        BatchPredictionId='ml-batch-id-%s' % random_id,
        BatchPredictionName='ml-batch-id-%s' % random_id,
        MLModelId='ml-model-id-%s' % random_id,
        BatchPredictionDataSourceId='ml-datasource-id-test-%s' % random_id,
        OutputUri='s3://%s' % ml_bucket_id,
    )

    while ml.get_batch_prediction(BatchPredictionId='ml-batch-id-%s' % random_id)['Status'] != 'COMPLETED':
        # print ml.get_batch_prediction(BatchPredictionId='ml-batch-id-%s' % random_id)
        # print "DEBUG: Testing model"
        time.sleep(8)

    # print ml.get_batch_prediction(BatchPredictionId='ml-batch-id-%s' % random_id)

    end_test = timer()
    print('Testing took %i Seconds' % (end_test - start_test) );

    s3.Bucket(ml_bucket_id).download_file('batch-prediction/result/ml-batch-id-%s-test.csv.gz' % random_id, '%s.gz' % result_csv)

    csvGzIn = gzip.GzipFile('%s.gz' % result_csv, 'rb')
    s = csvGzIn.read()
    csvGzIn.close()

    csvOut = file(result_csv, 'wb')
    csvOut.write(s)
    csvOut.close()

    os.remove('%s.gz' % result_csv)

    prediction_results = []

    with open(result_csv) as csv_result_file:
        result_reader = csv.reader(csv_result_file, delimiter=',', quotechar='"')
        header = next(result_reader)
        for row in result_reader:

            if force_model_type == 'classification':
                max_value = max(row)
                max_index = row.index(max_value)
                prediction_results.append(header[max_index])

            if force_model_type == 'regression':
                prediction_results.append(row[0])

    with open(test_csv, 'rb') as csv_test_file:
        test_csv_reader = csv.reader(csv_test_file, delimiter=',', quotechar='"', lineterminator="\n")
        test_csv_list = list(test_csv_reader)

    results = [test_csv_list[ix] + [prediction_results[ix]] for ix in range(len(prediction_results))]

    with open(result_csv, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(results)

    response = ml.delete_data_source(DataSourceId='ml-datasource-id-test-%s' % random_id)
    response = ml.delete_data_source(DataSourceId='ml-datasource-id-%s' % random_id)
    response = ml.delete_ml_model(MLModelId='ml-model-id-%s' % random_id)