import csv

with open(csv_file, 'rb') as csv_result_file:
    test_csv_reader = csv.reader(csv_result_file, delimiter=',', quotechar='"', lineterminator="\n")
    with open(csv_file + '.reordered', 'wb') as csv_test_file:
        result_csv_writer = csv.writer(csv_test_file, delimiter=',', quotechar='"', lineterminator="\n")
        for row in test_csv_reader:
            insert = row.pop()
            cleanded_row = row;
            cleanded_row = [column.strip() for column in cleanded_row]
            result_csv_writer.writerow([insert]  + cleanded_row)
