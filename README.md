🌾 Rice Classification using K-Means Clustering
===============================================

📌 Project Overview
-------------------

This project demonstrates unsupervised learning through **K-Means Clustering** on a rice classification dataset containing various physical characteristics of rice kernels. The primary goal is to discover hidden groupings and patterns within the dataset and visualize them effectively using Python libraries like `seaborn`, `matplotlib`, and `scikit-learn`.

* * *

📂 What’s Inside
----------------

*   Data Preprocessing
    
*   Exploratory Data Analysis (EDA)
    
*   Outlier Detection
    
*   Feature Selection
    
*   K-Means Clustering with Elbow Method
    
*   Cluster Visualization
    

* * *

🚀 Technologies Used
--------------------

*   ![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)    
*   ![Pandas](https://img.shields.io/badge/Pandas-2.1.4-150458?logo=pandas&logoColor=white)  
*   ![NumPy](https://img.shields.io/badge/NumPy-1.26.4-013243?logo=numpy&logoColor=white)              
*   ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8.4-11557C?logo=matplotlib&logoColor=white)  
*   ![Seaborn](https://img.shields.io/badge/Seaborn-0.13.2-42A5F5?logo=seaborn&logoColor=white)  
*   ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikit-learn&logoColor=white) 
*   ![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?logo=pytorch&logoColor=white)
*   PyTorch(for GPU setup)
    

* * *

📊 Data Exploration & Cleaning
------------------------------

*   Loaded the dataset from a zipped CSV file.
    
*   Verified that the dataset contains **no nulls or duplicates**.
    
*   Dropped irrelevant features like `id`, `Area`, and `Class` for clustering.
    
*   Checked feature relationships using:
    
    *   ✅ Correlation Heatmap
        
    *   ✅ Boxplots for outlier detection
        
    *   ✅ Histograms for distribution insights
        

📷 _Visuals Generated:_

*   📌 `Correlation Heatmap`: Shows how features interact with each other.
![corr](https://github.com/user-attachments/assets/94fae399-7e3b-495c-b85a-f3377425e6b4)
    
*   📌 `Boxplot`: Highlights the presence of outliers.
![boxplot](https://github.com/user-attachments/assets/eaf5733e-859c-424b-9e79-07af710f5c50)
    
*   📌 `Histogram Grid`: Displays feature-wise distributions.
![feature-distro](https://github.com/user-attachments/assets/1c97fc1e-47d7-4b52-99e3-53255080c7fd)
    

* * *

🤖 Clustering Workflow
----------------------

*   Applied **K-Means Clustering** with a loop from `k=2` to `k=20`.
    
*   Used the **Elbow Method** to find the optimal number of clusters.
    
*   Trained final model with `k=5`.
    

📷 _Elbow Curve_: Clearly visualizes the inflection point for optimal cluster count.
*   📌 `elbow grid`: Displays best value of k using elbow method.
![elbow](https://github.com/user-attachments/assets/b3328eb4-f60c-4c27-8b92-53a830eeebad)
* * *

🧠 Final Clustering
-------------------

*   Attached predicted cluster labels back to the original data.
    
*   Visualized the clusters using:
    
    *   ✅ 2D Scatter Plot (First two features plotted with color-coded clusters)
        

📷 _Cluster Visualization_: Easy to understand how data points are segmented.
* * *

📌 Final Output
*   📌 `cluster graph`: Displays clusters made after training.
![final](https://github.com/user-attachments/assets/064df2ba-97eb-438e-b298-0b46b75767f2)
---------------

After training with `k=5`, the model clustered the data into the following distribution:

*   Cluster 0: 5,415 samples
    
*   Cluster 1: 3,181 samples
    
*   Cluster 2: 1,920 samples
    
*   Cluster 3: 3,314 samples
    
*   Cluster 4: 4,355 samples
    

* * *

🚀 **Run on Google Colab**
--------------------------

You can explore and run this project directly on **[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/189EVx1JNIHh5Vx36zu9u-WnNX2KoCnVh#scrollTo=NI7u25pcg7jS)**:  

📥 **Note**:  
To use the dataset, upload the file manually from your local system:  
**`resources/data/riceclassification.csv`** into Colab after opening the notebook.  
You don't need a Kaggle API key or link for this project.


✅ Key Highlights
----------------

*   Fully GPU-supported with PyTorch integration 🔥
    
*   Clean, structured code with intuitive visualizations
    
*   Powerful unsupervised learning pipeline
    
*   Easy to extend for other datasets or clustering techniques
    

* * *

👨‍💻 Author
------------

**Muhammad Ahmad Naddem**  
* * *

🔥 _Clone it. Run it. Visualize the magic of unsupervised learning._  
🧠 _Great for ML beginners, data science students, and exploratory clustering analysis._