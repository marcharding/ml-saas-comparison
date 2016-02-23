from gcloud import storage
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from oauth2client.client import GoogleCredentials
from timeit import default_timer as timer
import csv
import gcloud
import hashlib
import httplib2
import json
import os
import random
import sys
import time

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

print('Training took %i Seconds.' % (end_training - start_training) ); 

# test model
start_test = timer()

print('Testing model.')
# batch processing seems to be broken, see https://github.com/google/oauth2client/issues/343
with open(result_csv, 'wb') as csv_result_file:
    result_csv_writer = csv.writer(csv_result_file, delimiter=',', quotechar='"', lineterminator="\n")
    with open(test_csv) as csv_test_file:
        test_csv_reader = csv.reader(csv_test_file, delimiter=',', quotechar='"')
        for row in test_csv_reader:
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
            try :
                result = request.execute()   
            except HttpError as err:
                print err
                if err.resp.status in [403, 500, 503]:
                  result = request.execute()           
                else: 
                    raise            
            result_csv_writer.writerow([result['outputLabel']] + row)

end_test = timer()

print('Testing took %i Seconds' % (end_test - start_test) ); 

existing_models = service.trainedmodels().list(project=project_id).execute()

for model in existing_models['items']:
    service.trainedmodels().delete(project=project_id,id=model['id']).execute()