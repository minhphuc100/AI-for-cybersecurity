from sklearn.metrics import (
    confusion_matrix, precision_score,
    recall_score, f1_score,
    classification_report, balanced_accuracy_score
)
 
# IDS results: 1=attack, 0=normal
y_true = [1,1,1,1,0,0,0,0,0,0]
y_pred = [1,1,0,1,0,0,1,0,1,0]
 
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
# TP=3 FP=2 FN=1 TN=4
 
print(f"Precision : {precision_score(y_true,y_pred):.2f}")
print(f"Recall    : {recall_score(y_true,y_pred):.2f}")
print(f"F1-Score  : {f1_score(y_true,y_pred):.2f}")
print(f"Bal.Acc   : {balanced_accuracy_score(y_true,y_pred):.2f}")
 
print(classification_report(
    y_true, y_pred,
    target_names=['Normal', 'Attack']))
