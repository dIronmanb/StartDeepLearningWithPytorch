from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def get_metrics(test, predicted, config):
    name = config['metrics']
    if name == "accuracy_score":
        return accuracy_score(test, predicted)
    elif name  == "recall_score":
        return recall_score(test, predicted)
    elif name == "precision_score":
        return precision_score(test, predicted)
    elif name == "f1_score":
        return f1_score(test, predicted)
    else:
        print("There is no metrics in metrics_name")