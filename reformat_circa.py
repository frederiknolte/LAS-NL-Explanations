import argparse
import pandas as pd
import random


def reformat_NLI(input_path, target_path, prediction_path, drop_none):
    possible_labels = {"neutral": 0, "entailment": 1, "contradiction": 2, "none": 3}

    with open(input_path, mode='r') as f:
        inputs = f.readlines()
    with open(target_path, mode='r') as f:
        targets = f.readlines()
    with open(prediction_path, mode='r') as f:
        predictions = f.readlines()

    data = []

    for line in zip(inputs, targets, predictions):
        hypothesis = line[0][26:-2].split(" premise: ")[0]
        premise = line[0][26:-2].split(" premise: ")[1]
        target = possible_labels[line[1][11:-23]]
        prediction = possible_labels[line[2][11:].split("', 'explanations': [")[0]]
        explanation = line[2][11:].split("', 'explanations': [")[1][1:-4]

        data.append([hypothesis, premise, target, prediction, explanation])

    random.shuffle(data)
    split_1 = int(len(data) * 0.6)
    split_2 = int(len(data) * 0.8)

    data = pd.DataFrame(data, columns=  ['hypothesis',
                                         'premise',
                                         'target',
                                         'prediction',
                                         'explanation'])

    if drop_none:
        data = data[data.target != 3]


    data.iloc[:split_1,:].to_csv('sim_experiments/data/circa/NLI/train.csv', sep=',', index=False)
    data.iloc[split_1:split_2,:].to_csv('sim_experiments/data/circa/NLI/dev.csv', sep=',', index=False)
    data.iloc[split_2:,:].to_csv('sim_experiments/data/circa/NLI/test.csv', sep=',', index=False)


def reformat_QA(input_path, target_path, prediction_path):
    possible_choices = {"Yes": 0,
                        "Yes, subject to some conditions": 1,
                        "No": 2,
                        "In the middle, neither yes nor no": 3,
                        "NA": 4,
                        "Other": 5}

    with open(input_path, mode='r') as f:
        inputs = f.readlines()
    with open(target_path, mode='r') as f:
        targets = f.readlines()
    with open(prediction_path, mode='r') as f:
        predictions = f.readlines()

    data = []

    for line in zip(inputs, targets, predictions):
        question = line[0][2:-2].split(" choice: ")[0]
        choices = possible_choices.keys()
        target = line[1][11:-23]
        prediction = possible_choices[line[2][11:].split("', 'explanations': [")[0].replace('"', '')]
        explanation = line[2][11:].split("', 'explanations': [")[1][1:-4]

        data.append([question, *choices, target, prediction, explanation])

    random.shuffle(data)
    split_1 = int(len(data) * 0.6)
    split_2 = int(len(data) * 0.8)

    data = pd.DataFrame(data, columns=['question',
                                       *[f"choice_{i}" for i in range(0, 6)],
                                       'target',
                                       'prediction',
                                       'explanation'])

    data.iloc[:split_1, :].to_csv('sim_experiments/data/circa/QA/train.csv', sep=',', index=False)
    data.iloc[split_1:split_2, :].to_csv('sim_experiments/data/circa/QA/dev.csv', sep=',', index=False)
    data.iloc[split_2:, :].to_csv('sim_experiments/data/circa/QA/test.csv', sep=',', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--QA', action='store_true', help='Flag for reformating QA data')
    parser.add_argument('--NLI', action='store_true', help='Flag for reformating NLI data')
    parser.add_argument('--drop_none', action='store_true', help='Flag for dropping all samples with target none')
    parser.add_argument("--target_path", required=True, default=None, type=str, help='Path of the target log file')
    parser.add_argument("--input_path", required=True, default=None, type=str, help='Path of the input log file')
    parser.add_argument("--prediction_path", required=True, default=None, type=str, help='Path of the prediction log file')
    parser.add_argument("--seed", default=0, type=int, help='Seed for train/dev/test split')
    args = parser.parse_args()

    random.seed(args.seed)

    if args.QA:
        reformat_QA(args.input_path, args.target_path, args.prediction_path)
    elif args.NLI:
        reformat_NLI(args.input_path, args.target_path, args.prediction_path, args.drop_none)
    else:
        raise Exception("Must select either QA or NLI option!")

