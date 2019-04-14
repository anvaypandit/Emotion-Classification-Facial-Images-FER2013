import pandas as pd
import os

def evaluate_class_accuracy(result_dir='Results',result_file_name='Results.csv'):
    fp = os.path.join(os.getcwd(),result_dir)
    fp = os.path.join(fp,result_file_name)
    df = pd.read_csv(fp)
    correct_pred = 0
    total_pred = 0

    for row in df.itertuples():
        total_pred += 1
        imageLabel = row.Filename.split('/')[1]
        trueLabel = int(imageLabel[0])
        if trueLabel == int(row.Predictions):
            correct_pred += 1
    accuracy = (correct_pred / total_pred)
    return accuracy


def test(result_file_name='Results.csv'):
    return evaluate_class_accuracy(result_file_name=result_file_name)


print(test('Results6123.csv'))