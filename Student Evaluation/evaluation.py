import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
pd.options.display.max_columns = 99

df = pd.read_csv('turkiye-student-evaluation_generic.csv')
# print(df.describe())
# print(df.info())


plt.style.use("fivethirtyeight")

# finding mean of questions
x_questions = df.iloc[:,5:33]
q_mean = x_questions.mean(axis=0)
total_mean = q_mean.mean()

q_mean = q_mean.to_frame('mean')
q_mean.reset_index(level=0, inplace=True)

# sns.barplot(x='index', y='mean', data=q_mean)
# plt.show()

cor = df.corr()
# plt.figure(figsize=(14,7))
# sns.heatmap(cor, annot=True, cmap='coolwarm')
# plt.show()

# PCA

x = df.iloc[:,5:33]
pca = PCA(n_components=2, random_state=42)
x_pca = pca.fit_transform(x)

# print(pca.explained_variance_ratio_.cumsum()[1]) 


# K-means cluster

distortions =[]
cluster_range = range(1,6)

# Elbow method
for i in cluster_range:
    model = KMeans(n_clusters = i, init ='k-means++', random_state=42)
    model.fit(x_pca)
    distortions.append(model.inertia_)

# plt.plot(cluster_range, distortions,marker = 'o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Distortions')
# plt.show()

model = KMeans(n_clusters = 3, init ='k-means++', random_state=42)
model.fit(x_pca)
y = model.predict(x_pca)

plt.scatter(x_pca[y==0, 0], x_pca[y==0, 1], s =50, c='red', label='cluster 1')
plt.scatter(x_pca[y==1, 0], x_pca[y==1, 1], s =50, c='blue', label='cluster 2')
plt.scatter(x_pca[y==2, 0], x_pca[y==2, 1], s =50, c='green', label='cluster 3')
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], s =100, c='yellow', label='centroids')
plt.title("Cluster of Students")
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()