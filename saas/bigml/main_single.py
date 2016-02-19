from bigml.api import BigML
import csv
import time

api = BigML(dev_mode=True)

# get args
train_csv = sys.argv[1]
test_csv = sys.argv[2]

# train model
source_train = api.create_source('./../../data/census/train.csv')
dataset_train = api.create_dataset(dataset_train)
model = api.create_model(dataset)

# test model
with open('./data/census/test.csv', 'rb') as csv_test_file:
    test_csv_reader = csv.reader(csv_test_file, delimiter=',', quotechar='"')
    for row in test_csv_reader:   
        row.pop()
        row = dict(zip(range(0, len(row)), row))
        prediction = api.create_prediction(model, row)
        api.pprint(prediction)