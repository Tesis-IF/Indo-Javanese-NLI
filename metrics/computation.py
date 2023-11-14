import numpy as np
import logging
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    labels = np.argmax(labels[:,0,:], axis=1)

    try:
        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred, average='micro')
        precision = precision_score(y_true=labels, y_pred=pred, average='micro')
        f1 = f1_score(y_true=labels, y_pred=pred, average='weighted')
    except Exception as e:
        logging.error("compute_metrics: ", e)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}