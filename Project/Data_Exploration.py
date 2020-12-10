import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="white", palette="muted", color_codes=True)

# df_ori = pd.read_csv('Feature_Matrix_GICU_with_missing_values.csv')
df_ori = pd.read_csv('Feature_Matrix_MIMIC_with_missing_values.csv')
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


# correlations between the variables
sns.pairplot(X)
plt.show()


# distributions of the variables
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
sns.despine(left=True)

sns.distplot(X, kde=False, color="b", ax=axes[0, 0])

# Plot a kernel density estimate and rug plot
sns.distplot(X, hist=False, rug=True, color="r", ax=axes[0, 1])

# Plot a filled kernel density estimate
sns.distplot(X, hist=False, color="g", kde_kws={"shade": True}, ax=axes[1, 0])

# Plot a historgram and kernel density estimate
sns.distplot(X, color="m", ax=axes[1, 1])

plt.tight_layout()
plt.show()


# outliers
sns.boxplot(X)
plt.show()

u = X.mean()  # calculate mean
std = X.std()  # calculate Standard deviation
print('Mean：', u)
print('Std', std)

# Box plot analysis
fig = plt.figure(figsize = (10,6))
ax1 = fig.add_subplot(2,1,1)
color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
X.plot.box(vert=False, grid = True,color = color,ax = ax1,label = 'sample')
# Box chart to see data distribution
# Bounded by inner limit

plt.show()