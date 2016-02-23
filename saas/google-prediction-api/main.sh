#!/bin/bash

python main.py './../../data/iris/train_first_field.csv' './../../data/iris/test_without_results.csv' './../../data/iris/google_prediciton_api_results.csv'

python main.py './../../data/census/train_first_field.csv' './../../data/census/test_without_results.csv' './../../data/census/google_prediciton_api_results.csv'

# diff -U 0 data/iris/test.csv  data/iris/result.csv | grep ^@ | wc -l