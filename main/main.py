import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
from scipy.stats import zscore
from sklearn.cluster import KMeans

'''
connect to the GPU
'''
if torch.cuda is not None:
  device = torch.device('cuda')
else:
  device = torch.device('cpu')
print(device)



'''
load the data
'''
zipf = r'/content/riceClassification.csv.zip'
with zipfile.ZipFile(zipf, 'r') as zipi:
  zipi.extractall()
  print('Done')

data = pd.read_csv(r'/content/riceClassification.csv')
print(data.head())
# id	Area	MajorAxisLength	MinorAxisLength	Eccentricity	ConvexArea	EquivDiameter	Extent	Perimeter	Roundness	AspectRation	Class
# 0	1	4537	92.229316	64.012769	0.719916	4677	76.004525	0.657536	273.085	0.764510	1.440796	1
# 1	2	2872	74.691881	51.400454	0.725553	3015	60.471018	0.713009	208.317	0.831658	1.453137	1
# 2	3	3048	76.293164	52.043491	0.731211	3132	62.296341	0.759153	210.012	0.868434	1.465950	1
# 3	4	3073	77.033628	51.928487	0.738639	3157	62.551300	0.783529	210.657	0.870203	1.483456	1
# 4	5	3693	85.124785	56.374021	0.749282	3802	68.571668	0.769375	230.332	0.874743	1.510000	1



data['Class'].value_counts()
# count Class	
# 1	  9985
# 0	  8200


data.describe()

data.info()
# RangeIndex: 18185 entries, 0 to 18184
# Data columns (total 12 columns):
#  #   Column           Non-Null Count  Dtype  
# ---  ------           --------------  -----  
#  0   id               18185 non-null  int64  
#  1   Area             18185 non-null  int64  
#  2   MajorAxisLength  18185 non-null  float64
#  3   MinorAxisLength  18185 non-null  float64
#  4   Eccentricity     18185 non-null  float64
#  5   ConvexArea       18185 non-null  int64  
#  6   EquivDiameter    18185 non-null  float64
#  7   Extent           18185 non-null  float64
#  8   Perimeter        18185 non-null  float64
#  9   Roundness        18185 non-null  float64
#  10  AspectRation     18185 non-null  float64
#  11  Class            18185 non-null  int64  
# dtypes: float64(8), int64(4)
# memory usage: 1.7 MB

'''
cleaning the data
'''

data.duplicated().sum().sum()
# np.int64(0)

data.isnull().sum().sum()
# np.int64(0)

'''visualization'''

# correlation
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

#outliers
plt.figure(figsize=(18, 3))
sns.boxplot(data=data, palette="Set2")
plt.xticks(rotation=45)
plt.title("Boxplot of Features")
plt.show()

#feature distribution
data.hist(figsize=(12, 12), bins=20, color='skyblue', edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

'''
droping the data
'''

data = data.drop(columns=['id', 'Area', 'Class'])

'''
training the model
'''
intertia = []
# k_range = (2,21)
for k in range(2,21):
  km = KMeans(n_clusters=k, init='k-means++')
  km.fit(data)
  intertia.append(km.inertia_)

'''
plotting the elbow curve
'''

plt.figure(figsize=(10, 6))
plt.plot([i for i in range (2,21)], intertia, color='blue')
plt.xlabel('Number of Clusters')
plt.xticks([i for i in range (2,21)])
plt.ylabel('Intertia')
plt.title('The Elbow Method')
plt.grid(True)
plt.show()


'''
training the model with the best k
'''
k = 5
kmc = KMeans(n_clusters=k, init='k-means++', random_state=42)
numeric_data = data.select_dtypes(include=['number'])
# Get cluster labels
cluster_labels = kmc.fit_predict(numeric_data)
# Assign the labels back to the original DataFrame
data['classs'] = cluster_labels


'''verifing'''
data['classs'].value_counts()
# classs count	
# 0	    5415
# 4	    4355
# 3	    3314
# 1	    3181
# 2	    1920


'''visualizing the clusters'''
plt.figure(figsize=(8, 5))
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=data['class'], cmap='viridis')
plt.title('Clusters Visualization (First Two Features)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()

