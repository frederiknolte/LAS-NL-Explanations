import random
import argparse
import pandas as pd
import os
from google.cloud import storage


def read_bucket_files(bucket_name, rs, mu, seed, step, drop_none, random_seed):
    possible_labels = {"neutral": 0, "entailment": 1, "contradiction": 2, "none": 3}

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    validation_data = []
    # READ VALIDATION SET
    validation_prediction_prefix = f"esnli_and_cos_e_to_circa_nli_{rs}_{mu}{str(seed)}/validation_eval/circa_eval_v100_nli_{rs}" \
                                   f"_{mu}" \
                                   f"{str(seed)}_{step}_predictions"
    validation_input_prefix = f"esnli_and_cos_e_to_circa_nli_{rs}_{mu}{str(seed)}/validation_eval/circa_eval_v100_nli_{rs}" \
                              f"_{mu}" \
                              f"{str(seed)}_inputs"
    validation_target_prefix = f"esnli_and_cos_e_to_circa_nli_{rs}_{mu}{str(seed)}/validation_eval/circa_eval_v100_nli_{rs}" \
                               f"_{mu}" \
                               f"{str(seed)}_targets"
    validation_prediction_blobs = list(bucket.list_blobs(prefix=validation_prediction_prefix))
    validation_input_blobs = list(bucket.list_blobs(prefix=validation_input_prefix))
    validation_target_blobs = list(bucket.list_blobs(prefix=validation_target_prefix))
    validation_prediction = validation_prediction_blobs[0].download_as_string()
    validation_input = validation_input_blobs[0].download_as_string()
    validation_target = validation_target_blobs[0].download_as_string()
    validation_prediction = validation_prediction.decode('utf-8')
    validation_input = validation_input.decode('utf-8')
    validation_target = validation_target.decode('utf-8')

    for line in zip(validation_input.splitlines(), validation_target.splitlines(), validation_prediction.splitlines()):
        hypothesis = line[0][26:-2].split(" premise: ")[0]
        premise = line[0][26:-1].split(" premise: ")[1]
        target = possible_labels[line[1][11:-22]]
        prediction = possible_labels[line[2][11:].split("', 'explanations': [")[0]]
        explanation = line[2][11:].split("', 'explanations': [")[1][1:-3]
        validation_data.append([hypothesis, premise, target, prediction, explanation])

    test_data = []
    # READ TEST SET
    test_prediction_prefix = f"esnli_and_cos_e_to_circa_nli_{rs}_{mu}{str(seed)}/test_eval/circa_eval_v100_nli_{rs}" \
                             f"_{mu}" \
                             f"{str(seed)}_{step}_predictions"
    test_input_prefix = f"esnli_and_cos_e_to_circa_nli_{rs}_{mu}{str(seed)}/test_eval/circa_eval_v100_nli_{rs}" \
                        f"_{mu}" \
                        f"{str(seed)}_inputs"
    test_target_prefix = f"esnli_and_cos_e_to_circa_nli_{rs}_{mu}{str(seed)}/test_eval/circa_eval_v100_nli_{rs}" \
                         f"_{mu}" \
                         f"{str(seed)}_targets"
    test_prediction_blobs = list(bucket.list_blobs(prefix=test_prediction_prefix))
    test_input_blobs = list(bucket.list_blobs(prefix=test_input_prefix))
    test_target_blobs = list(bucket.list_blobs(prefix=test_target_prefix))
    test_prediction = test_prediction_blobs[0].download_as_string()
    test_input = test_input_blobs[0].download_as_string()
    test_target = test_target_blobs[0].download_as_string()
    test_prediction = test_prediction.decode('utf-8')
    test_input = test_input.decode('utf-8')
    test_target = test_target.decode('utf-8')

    for line in zip(test_input.splitlines(), test_target.splitlines(), test_prediction.splitlines()):
        hypothesis = line[0][26:-2].split(" premise: ")[0]
        premise = line[0][26:-1].split(" premise: ")[1]
        target = possible_labels[line[1][11:-22]]
        prediction = possible_labels[line[2][11:].split("', 'explanations': [")[0]]
        explanation = line[2][11:].split("', 'explanations': [")[1][1:-3]
        test_data.append([hypothesis, premise, target, prediction, explanation])

    # random.seed(random_seed)
    # random.shuffle(data)

    validation_data = pd.DataFrame(validation_data, columns=['hypothesis',
                                       'premise',
                                       'target',
                                       'prediction',
                                       'explanation'])

    test_data = pd.DataFrame(test_data, columns=['hypothesis',
                                       'premise',
                                       'target',
                                       'prediction',
                                       'explanation'])

    if drop_none:
        validation_data = validation_data[validation_data.target != 3]
        test_data = test_data[test_data.target != 3]
        # validation_data = validation_data[validation_data.target != 3]

    # split_1 = int(len(data) * 0.6)
    # split_2 = int(len(data) * 0.8)

    os.makedirs("data/circa/NLI/", exist_ok=True)
    test_data.to_csv('data/circa/NLI/train.csv', sep=',', index=False)
    validation_data.to_csv('data/circa/NLI/dev.csv', sep=',', index=False)
    # data.iloc[split_2:, :].to_csv('data/circa/NLI/test.csv', sep=',', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket_name", default = None, type=str, help="Name of the GCloud bucket to read data from. If None, no data is read from GCloud")
    parser.add_argument("--rs", default=None, type=str, choices=["relaxed", "strict"],
                        help="Whether to use relaxed or strict data")
    parser.add_argument("--mu", default=None, type=str, choices=["matched", "unmatched"],
                        help="Whether to use matched or unmatched data")
    parser.add_argument("--gcloud_seed", default=None, type=int, help="Seed that was used to obtain data in GCloud bucket")
    parser.add_argument("--random_seed", default=21, type=int, help="Seed that is used to shuffle the data")
    parser.add_argument("--gcloud_step", default=None, type=int, help="Seed that was used to obtain data in GCloud bucket")
    parser.add_argument('--drop_none', action='store_true', default=False, help='Drop none from data when reading from GCloud bucket')

    args = parser.parse_args()
    read_bucket_files(args.bucket_name, args.rs, args.mu, args.gcloud_seed, args.gcloud_step, args.drop_none, args.random_seed)