import pandas as pd
from Curve import plot_pr_curve
from Curve import plot_roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn import metrics

df_ori = pd.read_csv('Feature_Matrix_GICU_with_missing_values.csv')
# df_ori = pd.read_csv('Feature_Matrix_MIMIC_with_missing_values.csv')
# Load data

df_new = df_ori.drop(['cohort'], axis=1)

df_new.reset_index()
df_new.set_index('ICUSTAY_ID')
# Set ICUSTAY_ID as the index. This way it will not be used as a feature column.

df_new = df_new.dropna()
# print(np.isnan(df_new).any())
# Remove rows with missing data.
features = ['creatinine', 'po2', 'fio2', 'pco2', 'bp_min', 'bp_max', 'pain', 'k', 'hr_min', 'hr_max', 'gcs_min',
            'gcs_max',
            'bun', 'hco3', 'airway', 'resp_min', 'resp_max', 'haemoglobin', 'spo2_min', 'spo2_max', 'temp_min',
            'temp_max',
            'na']

X = df_new[features]
y = df_new.outcome

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

logistic = LogisticRegression()
# Scikitlearn's Logistic classifier:
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html


print('Logistic Score: %f' % logistic.fit(X_train, y_train).score(X_test, y_test))

y_score = logistic.predict_proba(X_test)
# print(y_score)

fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
# fpr, tpr, _ = roc_curve(y_test, y_score)
# fpr:假正率 tpr：召回率

precision, recall, thresholds = precision_recall_curve(y_test, y_score[:, 1])
# P-R value

# print("fpr:",fpr)
# print("tpr:", tpr)
print("recall:", recall)
print("precision:", precision)

plot_roc_curve(fpr, tpr)
plot_pr_curve(precision, recall)
print('Logistic ROC auc: %f' % auc(fpr, tpr))
print('Logistic PR auc: %f' % auc(recall, precision))
print('Logistic PR ap: %f' % metrics.average_precision_score(y_test, y_score[:, 1]))
