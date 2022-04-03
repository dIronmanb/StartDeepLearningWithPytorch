'''
 정확도 평가 지표 정보
 accuracy:
 recall:
 precision:
 f1:
 confusion_matrix:


 prdicted set과 target set에 대해 판별(인자는 list..???)
'''


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix


def get_metrics(targets, predicted, config):
    name = config['metrics']
    if name == "accuracy_score":
        return accuracy_score(targets, predicted)
    elif name  == "recall_score":
        return recall_score(targets, predicted)
    elif name == "precision_score":
        return precision_score(targets, predicted)
    elif name == "f1_score":
        return f1_score(targets, predicted)
    elif name == "confusion_matrix":
        return confusion_matrix
    else:
        print("There is no metrics in metrics_name")
        
        
def get_confusion_metric(targets, predicted):
    return confusion_matrix(targets, predicted)

def get_recall_score(targets, predicted):
    return recall_score(targets, predicted)

def get_precision_score(targets, predicted):
    return precision_score(targets, predicted)

