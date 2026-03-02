from typing import Dict, Optional

import numpy as np
from sklearn.metrics import confusion_matrix

def print_confusion_matrix(
    y_true,
    y_pred,
    *,
    id2label: Optional[Dict[int, str]] = None,
    normalize: bool = True,
    digits: int = 3,
):
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype(np.float32)
        cm = cm / cm.sum(axis=1, keepdims=True)

    if id2label is not None:
        labels = [id2label[i] for i in range(len(id2label))]
    else:
        labels = [str(i) for i in range(cm.shape[0])]

    max_label_len = max(len(l) for l in labels)

    header = " " * (max_label_len + 2)
    for lbl in labels:
        header += f"{lbl:>{max_label_len+2}}"
    print(header)

    for i, row in enumerate(cm):
        row_str = f"{labels[i]:>{max_label_len}} |"
        for val in row:
            if normalize:
                row_str += f"{val:>{max_label_len+2}.{digits}f}"
            else:
                row_str += f"{int(val):>{max_label_len+2}d}"
        print(row_str)

    print()    
