import os
import sys

import mlflow
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from src import MODELS_DIR#, SFTP


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def log_artifact(path, artifact_path=None, is_remote=False):
    if not is_remote:
        mlflow.log_artifact(path, artifact_path)
    else:
        remote_path = f'{mlflow.get_artifact_uri()}/{artifact_path}'
        try:
            SFTP.chdir(remote_path)  # Test if remote_path exists
        except IOError:
            SFTP.mkdir(remote_path)
            SFTP.chdir(remote_path)

        SFTP.put(path, f'{remote_path}/{os.path.basename(path)}')


def save_model(model, log_to_mlflow=False, is_remote=False, epoch=''):
    path = f'{MODELS_DIR}/{epoch}{model.__class__.__name__}.pth'
    torch.save(model, path)

    if log_to_mlflow:
        log_artifact(path, artifact_path='model', is_remote=is_remote)
        os.remove(path)


def calculate_metrics(scores: np.array, true_labels: np.array, score_name: str, verbose=True):
    # ROC-AUC & APS
    roc_auc = roc_auc_score(true_labels, scores)
    aps = average_precision_score(true_labels, scores)

    # Mean score on validation
    mean_score = scores.mean()

    # F1-score & optimal threshold
    # if opt_threshold is None:  # validation
    #     precision, recall, thresholds = precision_recall_curve(y_true=true_labels, probas_pred=scores)
    #     f1_scores = (2 * precision * recall / (precision + recall))
    #     f1 = np.nanmax(f1_scores)
    #     opt_threshold = thresholds[np.nanargmax(f1_scores)]
    # else:  # testing
    #     y_pred = (scores > opt_threshold).astype(int)
    #     f1 = f1_score(y_true=true_labels, y_pred=y_pred)

    if verbose:
        print(f'ROC-AUC on {score_name}: {roc_auc}. APS on {score_name}: {aps}. Mean {score_name}: {mean_score}')
    # print(f'F1-score on {type}: {f1}. Optimal threshold on {type}: {opt_threshold}')

    return {f"roc-auc_{score_name}": roc_auc,
            f"aps_{score_name}": aps,
            f"mean_{score_name}": mean_score}
