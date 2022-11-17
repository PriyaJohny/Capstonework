---
title: "Predicting Stroke "
author: "Priya Johny"
date: "today"

# date: "`r Sys.Date()`"

output:
  html_document:
    code_folding: hide
    # number_sections: true
    toc: yes
    toc_depth: 3
    toc_float: yes
  pdf_document:
    toc: yes
    toc_depth: '3'
---

```{r basic, include = F}
# use this function to conveniently load libraries and work smoothly with knitting
# can add quietly=T option to the require() function
# the loadPkg function essentially replaced/substituted two functions install.packages() and library() in one step.
loadPkg = function(x) { if (!require(x,character.only=T, quietly =T)) { install.packages(x,dep=T,repos="http://cran.us.r-project.org"); if(!require(x,character.only=T)) stop("Package not found") } }
# unload/detact package when done using it
unloadPkg = function(pkg, character.only = FALSE) { 
  if(!character.only) { pkg <- as.character(substitute(pkg)) } 
  search_item <- paste("package", pkg,sep = ":") 
  while(search_item %in% search()) { detach(search_item, unload = TRUE, character.only = TRUE) } 
}
```
# %%%%%%%%%%%%% Machine Learning %%%%%%%%%%%%%%%%%%%%%%%%

# Importing the required packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as ss
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score, recall_score, precision_score, average_precision_score, f1_score, classification_report, accuracy_score, plot_roc_curve, plot_precision_recall_curve, plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix,roc_auc_score, roc_curve, auc, precision_recall_curve
import eli5
from sklearn.impute import KNNImputer
from dtreeviz.trees import *
import graphviz
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
pd.options.mode.chained_assignment = None
sns.set_theme(style="whitegrid")


#%%-----------------------------------------------------------------------
#Load Dataset
df = pd.read_csv(r"C:\Users\ABC\Desktop\healthcare-dataset-stroke-data.csv")
#%%-----------------------------------------------------------------------
#Important Functions
def do_a_crosstab(col1, col2):
    plt.figure(figsize = (10, 11))
    plt.rcParams["figure.figsize"] = (10,12)
    tab = pd.crosstab(df[col1], df[col2],margins = True).sort_values('All',ascending=False)
    tab = tab.drop('All',axis=1)
    tab = tab.drop('All',axis=0)
    tab.plot(kind='bar', stacked = False)
    plt.show()
    
    
#This returns correlation between 0-1 for categorical features
def calculate_cramers_v(x, y):
    # plotting confusion matrix
    confusion_matrix = pd.crosstab(x,y)
    # finding chi_score
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

def cramers_corrected_stat_for_heatmap(confusion_matrix):
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
#%%-----------------------------------------------------------------------
# We need to drop the id column, its redundant an id column can never have a significant influence on the predition

df.drop("id", axis = 1, inplace = True)

df.head()
df.shape
df.describe()
df.isnull().sum()
#%%-----------------------------------------------------------------------
# Lets check the distribution of bmi column

sns.histplot(df["bmi"], kde = True);
plt.title("BMI Distrbution");

#The peak lies around 30.
#This is a right skewed plot.
#mean > mode > median
#Assumption: BMI is dependednt on gender and age, so lets fill the missing values for BMI on the basis of these
#%%-----------------------------------------------------------------------


#%%-----------------------------------------------------------------------
#%%-----------------------------------------------------------------------
#%%-----------------------------------------------------------------------
#%%-----------------------------------------------------------------------
#%%-----------------------------------------------------------------------
#%%-----------------------------------------------------------------------
#%%-----------------------------------------------------------------------
#%%-----------------------------------------------------------------------
#%%-----------------------------------------------------------------------
#%%-----------------------------------------------------------------------
#%%-----------------------------------------------------------------------
#%%-----------------------------------------------------------------------
#%%-----------------------------------------------------------------------
#%%-----------------------------------------------------------------------
#%%-----------------------------------------------------------------------
#%%-----------------------------------------------------------------------
#%%-----------------------------------------------------------------------
#%%-----------------------------------------------------------------------
#%%-----------------------------------------------------------------------
#%%-----------------------------------------------------------------------
#%%-----------------------------------------------------------------------
#%%-----------------------------------------------------------------------
#%%-----------------------------------------------------------------------
#%%-----------------------------------------------------------------------
#%%-----------------------------------------------------------------------





























## References: Stroke Prediction Dataset: Data file Retrieved from https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
