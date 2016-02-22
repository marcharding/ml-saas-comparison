from gcloud import storage
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from oauth2client.client import GoogleCredentials
from oauth2client.service_account import ServiceAccountCredentials
from timeit import default_timer as timer
import csv
import gcloud
import hashlib
from httplib2 import Http
import json
import os
import random
import sys
import time
from functools import partial

# get args
train_csv = sys.argv[1]
test_csv = sys.argv[2]
result_csv = sys.argv[3]

# project
with open(os.environ['GOOGLE_APPLICATION_CREDENTIALS']) as app_credentials_json:    
    app_credentials = json.load(app_credentials_json)
    project_id = app_credentials['project_id']

# model
model_id = "model-id-%i" % random.randint(0, 65535)

# bucket 
bucket_id = "bucket-id-%i" % random.randint(0, 65535)

# authenticate
client = storage.Client()
credentials = GoogleCredentials.get_application_default()

# instantiate service
service = build('prediction', 'v1.6', credentials=credentials)

try:
    bucket = client.get_bucket(bucket_id)
except gcloud.exceptions.NotFound:
    bucket = client.create_bucket(bucket_id)  

blob = bucket.blob('train.csv')
blob.upload_from_filename(filename=train_csv)

# train model
start_training = timer()

service.trainedmodels().insert(
    project = project_id,
    body = {
        'storageDataLocation': bucket_id + '/train.csv',
        'id': model_id,
    }    
).execute();

print('Training model.')
while service.trainedmodels().get(project=project_id, id=model_id).execute()['trainingStatus'] != 'DONE':
    # print "DEBUG: Training model"
    time.sleep(1)

end_training = timer()

# get model type
trained_model = service.trainedmodels().get(project=project_id, id=model_id).execute()
model_type = trained_model['modelInfo']['modelType']

print('Training took %i Seconds.' % (end_training - start_training) ); 

# test model
start_test = timer()

def batch_callback( row, results, position, model_type, request_id, response, exception):
  if exception is not None:
    print exception
    pass
  else:
    if model_type == 'regression':
        results.insert(position, [response['outputValue']] + row)
    else:
        results.insert(position, [response['outputLabel']] + row)
    pass

print('Testing model.')

# batch processing seems to be broken, took a while to figure this out
# see the differnece with the service account authentication
# credentials = GoogleCredentials.get_application_default() will NOT work
# see https://github.com/google/oauth2client/issues/343
# and https://github.com/google/google-api-python-client/issues/137
# reinstantiate service with working credentials for batch
credentials = ServiceAccountCredentials.from_json_keyfile_name(os.environ['GOOGLE_APPLICATION_CREDENTIALS'], ['https://www.googleapis.com/auth/prediction'])
http = credentials.authorize(Http())
credentials.refresh(http) 
service = build('prediction', 'v1.6', http=http)

i = 0
results = []
# reset results
open(result_csv, 'w').close()
with open(test_csv) as csv_test_file:
    test_csv_reader = csv.reader(csv_test_file, delimiter=',', quotechar='"')
    for row in test_csv_reader:

        # only 1000 requests/batch, see https://cloud.google.com/prediction/docs/reference/v1.6/batch
        # because the api is very flaky we just use 100
        # see https://github.com/google/google-api-ruby-client/issues/210#issuecomment-100377192

        if i % 100 == 0:
            batch = service.new_batch_http_request()
        
        # print "DEBUG: Testing model"
        request = service.trainedmodels().predict(
            project = project_id,
            id = model_id,
            body = {
            'input': {
                'csvInstance': 
                    row
                }
            }
        )  
        
        # callback with j to keep initla order due to async nature of batch callbacks
        partial_callback = partial(batch_callback, row, results, i, model_type)
        batch.add(request, callback=partial_callback)

        i = i+1
        if i % 100 == 0:
            print i
            batch.execute()
            # also only 100 requests per 100 seconds, so sleep for 2 seconds
            time.sleep(2)


# execute open batches
if i % 100 != 0:
    batch.execute()

# write results
with open(result_csv, 'wb') as csv_result_file:
    result_csv_writer = csv.writer(csv_result_file, delimiter=',', quotechar='"', lineterminator="\n")
    for row in results:
        result_csv_writer.writerow(row)

end_test = timer()

print('Testing took %i Seconds' % (end_test - start_test) ); 

existing_models = service.trainedmodels().list(project=project_id).execute()

for model in existing_models['items']:
    service.trainedmodels().delete(project=project_id,id=model['id']).execute()