---
title: " Stroke Analysis "
title: " Capstone "
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
# %%%%%%%%%%%%% Data Mining & Machine Learning %%%%%%%%%%%%%%%%%%%%%%%%

# Importing the required packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as ss
import itertools #effiecient looping 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #Standardization of datasets is a common requirement for many ML estimators
from imblearn.over_sampling import SMOTE #SMOTE: Synthetic Minority Over-sampling Technique
from collections import Counter #counting hashable object
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
df.shape
#%%-----------------------------------------------------------------------
df.info()
#%%-----------------------------------------------------------------------
df.head()
#%%-----------------------------------------------------------------------
df.describe()
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

#%%-----------------------------------------------------------------------

df.describe()

>We can see the min age is 0.08 indicating an infant and a max of 82 yr old adult involved in this dataset. 

>We see that mean of age is over 43yrs old, mean of glucose level falls on 106.14 with a bmi of mean 29.

>Classification refers to a predictive modeling problem where a class label is predicted, here stoke.

#%%-----------------------------------------------------------------------

df.isnull().sum()
#%%-----------------------------------------------------------------------
# Lets check the distribution of bmi column

ax = sns.histplot(df["bmi"], kde = True,color='teal',edgecolor='black');
ax.lines[0].set_color('crimson');
plt.title("BMI Distrbution");

> The peak lies around 30. 

> This is a right skewed plot("positively skewed" distribution.) 

> Mean > Median > Mode Therefore, here the value of mean is the most important.

> #Assumption: BMI is dependent on gender and age, so filling the missing values for BMI on the basis of these.
#%%-----------------------------------------------------------------------
# We would have to fill them with mean at gender and age
df['bmi'] = df.groupby(['gender', 'age'])['bmi'].transform(
    lambda group: group.fillna(np.mean(group))
)
df.isnull().sum()
#%%-----------------------------------------------------------------------
df[df['bmi'].isnull()]
#Even after filling out the missing values with mean, 1 value still is empty. Which means for this age and gender no other value exists.
#So lets just fill it on the basis of gender that seems more appropriate.
#%%-----------------------------------------------------------------------
# We would have to fill them with mean at gender
df['bmi'] = df.groupby(['gender'])['bmi'].transform(
    lambda group: group.fillna(np.mean(group))
)
#%%-----------------------------------------------------------------------
# No null value exists in our data now
df.isnull().sum()
#%%-----------------------------------------------------------------------
sns.countplot(x=df["gender"],color='teal'  ,edgecolor='black')
plt.title("Gender Vs Frequency count");
#%%-----------------------------------------------------------------------
sns.countplot(x=df["ever_married"],color='tab:olive'  ,edgecolor='black')
plt.title("Married Vs Frequency count");

# Married people nearly twice then unmarried people.

#%%-----------------------------------------------------------------------
sns.countplot(x=df["hypertension"],color='tab:orange', edgecolor='black')
plt.title("Hypertension Vs Frequency count");

#People having no hypertension is a lot more than the other class.

#%%-----------------------------------------------------------------------

sns.countplot(x=df["heart_disease"],color='yellow', edgecolor='black')
plt.title("Heart Disease Vs Frequency count");

#%%-----------------------------------------------------------------------

sns.countplot(x=df["work_type"],edgecolor='black')
plt.title("Type Of Work Vs Frequency count");


#%%-----------------------------------------------------------------------
sns.countplot(x=df["Residence_type"],color='tab:blue',edgecolor='black')
plt.title("Type Of Residence Vs Frequency count");


#%%-----------------------------------------------------------------------

sns.countplot(x=df["smoking_status"], color='blue', edgecolor='black')
plt.title("Smoking Status Vs Frequency count");

#%%-----------------------------------------------------------------------

# Lets check the distribution of age column
ax=sns.histplot(df["avg_glucose_level"], kde = True,color='teal',edgecolor='black');
ax.lines[0].set_color('crimson');
plt.title("Average Glucose Level Distrbution");

#%%-----------------------------------------------------------------------

# Lets check the distribution of age column
ax=sns.histplot(df["age"], kde = True, edgecolor='black');
ax.lines[0].set_color('crimson');
plt.title("Age Distrbution");

> Age distribution is between approx 0 to approx 80+. 

> The mode comes around 50-60.

#%%-----------------------------------------------------------------------

### People suffering from stroke, do they suffer from hypertension? 

#do_a_crosstab("stroke", "hypertension")
sns.countplot(x=df['stroke'], hue=df['hypertension'],color='maroon',edgecolor='black');
plt.title("People suffering from stroke, do they suffer from hypertension?");
# It can be observed people having hypertension suffered less from a heart stroke than those not having hypertension.

#%%-----------------------------------------------------------------------

### People suffering from stroke, do they have prior heart disease? 

#do_a_crosstab("stroke", "heart_disease")
sns.countplot(x=df['stroke'], hue=df['heart_disease'],color='green',edgecolor='black');
plt.title("People suffering from stroke, do they have prior heart disease?");

# It can be observed people having prior heart disease suffered less from a heart stroke than those not having hypertension.

#%%-----------------------------------------------------------------------

###Can work type of a person have impact on heart stroke?

#do_a_crosstab("stroke", "work_type")
sns.countplot(x=df['stroke'], hue=df['work_type'],color='red',edgecolor='black');
plt.title("Can work type of a person have impact on stroke?");
#Interesting thing to notice here is, "children" and "people who never worked" has almost 0 numbers for a heart stroke. This could mean heart attack is influenced by the type of work of a person.

#%%-----------------------------------------------------------------------

###What are the smoking patterns for married/unmarried people?

#do_a_crosstab("ever_married", "smoking_status")
sns.countplot(x=df['ever_married'], hue=df['smoking_status'],color='teal',edgecolor='black');
plt.title("What are the smoking patterns for married/unmarried people?");

#A large chunk of married people have never smoked.

#%%-----------------------------------------------------------------------
## Correlation check on stroke
#simple correlation with respect to stroke column
print(corr["stroke"])

sns.pairplot(df,hue='stroke',diag_kws={'bw':0.2} ).fig.suptitle("Stoke Pair Graph", y=1)
#%%-----------------------------------------------------------------------
## Cramer V Correlation between Categorical Variables
# Getting out the categorical variables
categorical_cols = [col for col in df.columns if df[col].dtype == "object"]


for col in df.columns:
    if df[col].nunique()==2:
        categorical_cols.extend([col])

continuous_cols = list(set(df.columns) - set(categorical_cols))


#%%-----------------------------------------------------------------------

for col1 in range(len(categorical_cols)):
    for col2 in range(col1+1, len(categorical_cols)):
        print(f"Corelation between column {categorical_cols[col1]} and {categorical_cols[col2]} is:  ", calculate_cramers_v(df[categorical_cols[col1]], df[categorical_cols[col2]]))

#%%-----------------------------------------------------------------------

correlation_matrix = np.zeros((len(categorical_cols),len(categorical_cols)))

for column1, column2 in itertools.combinations(categorical_cols, 2):
    index1, index2 = categorical_cols.index(column1), categorical_cols.index(column2)
    correlation_matrix[index1, index2] = cramers_corrected_stat_for_heatmap(pd.crosstab(df[column1], df[column2]))
    correlation_matrix[index2, index1] = correlation_matrix[index1, index2]
    
corr = pd.DataFrame(correlation_matrix, index = categorical_cols, columns = categorical_cols)

for col in ((corr.columns)):
    corr.loc[col, col] = 1


fig, ax = plt.subplots(figsize=(15, 12))

ax = sns.heatmap(corr, annot=True, ax=ax); ax.set_title("Cramer V Correlation between Categorical Variables");

# Nothing exceeds .70, so no point in dropping anything. As no strong correlation exists here.

#%%-----------------------------------------------------------------------

## Correlation check for continuous columns

fig, ax = plt.subplots(figsize=(7, 5))

ax = sns.heatmap(df[continuous_cols].corr(), annot=True, ax=ax); ax.set_title("Correlation plot between Continuous Variables");

#Nothing exceeds .70, so no point in dropping anything. As no strong correlation exists here.

#%%-----------------------------------------------------------------------
##Checks imbalance

def checks_imbalance(df, y_col):
    df[y_col].value_counts().plot(kind = 'bar')
    plt.title("Class distribution");
    plt.xlabel("Classes");
    plt.ylabel("Number of datapoints");
    plt.show()
    
    if df[y_col].value_counts()[0] > df[y_col].value_counts()[1]:
        print("Class imbalance exists.", end = ' ')
        print("Class", df[y_col].value_counts().index[0], "has more number of data-points. Total datapoints: ", df[y_col].value_counts()[0])
        return True
    
    elif df[y_col].value_counts()[0] == df[y_col].value_counts()[1]:
        print("No class imbalance exists.")
        return False

    
    else:
        print("Class imbalance exists.", end = ' ')
        print("Class", df[y_col].value_counts().index[1], "has more number of data-points. Total datapoints: ", df[y_col].value_counts()[1])
        return True

#%%-----------------------------------------------------------------------

checks_imbalance(df, "stroke")

# Huge imbalance exists. We need to deal with this using some balancing technique.

#%%-----------------------------------------------------------------------X

data = pd.get_dummies(df)

#%%-----------------------------------------------------------------------
## Modelling1

# Importing Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB

# Importing Evaluation matrces
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix


# check the performance on diffrent regressor
models = []
models.append(('XGBClassifier', XGBClassifier()))
models.append(('LogisitcRegression', LogisticRegression()))
models.append(('KNeighborsClassifier', KNeighborsClassifier()))
models.append(('RandomForestClassifier', RandomForestClassifier()))
models.append(('AdaBoostClassifier', AdaBoostClassifier()))
models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))


# metrices to store performance
acc = []
pre = []
f1 = []
con = []
rec = []


import time
i = 0
for name,model in models:
    i = i+1
    start_time = time.time()
    
    # Fitting model to the Training set
    clf = model
    clf.fit(X_train, y_train)
    
    # predict values
    y_pred = clf.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    acc.append(accuracy)
    # Precision
    precision = precision_score(y_test, y_pred, average=None)
    pre.append(precision)
    # Recall
    recall = recall_score(y_test, y_pred, average=None)
    rec.append(recall)
    # F1 Score
    f1_sco = f1_score(y_test, y_pred, average=None)
    f1.append(f1_sco)
    # Confusion Matrix
    confusion_mat = confusion_matrix(y_test, y_pred)
    con.append(confusion_mat)
    # Report
    report = classification_report(y_test, y_pred)


    print("+","="*100,"+")
    print('\033[1m' + f"\t\t\t{i}-For {name} the performance result is: " + '\033[0m')
    print("+","="*100,"+")
    print('Accuracy : ', accuracy)   
    print("-"*50)
    print('F1 : ', f1_sco)
    print("-"*50)
    print('Reacll : ', recall)
    print("-"*50)
    print('Precision : ', precision)
    print("-"*50)
    print('Confusion Matrix....\n', confusion_mat)
    print("-"*50)
    print('Classification Report....\n', report)
    print("-"*50)
    print('Plotting Confusion Matrix...\n')
    plot_confusion_matrix(clf, X_test, y_test)
    plt.show()


    
    print("\t\t\t\t\t\t\t-----------------------------------------------------------")
    print(f"\t\t\t\t\t\t\t Time for detection ({name}) : {round((time.time() - start_time), 3)} seconds...")
    print("\t\t\t\t\t\t\t-----------------------------------------------------------")
    print()

    #%%-----------------------------------------------------------------------
comp = pd.DataFrame({"Model": dict(models).keys(), "Accuracy": acc, "Precision": pre, "Recall": rec, "F1_Score": f1, "Confusion Matrix": con})
comp
#%%-----------------------------------------------------------------------

#Test and Train dataset
y = data["stroke"]
X = data[list(set(data.columns) - set(["stroke"]))]

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size = 0.2)

#%%-----------------------------------------------------------------------

X_train.reset_index(drop = True, inplace = True)
y_train.reset_index(drop = True, inplace = True)
X_test.reset_index(drop = True, inplace = True)
y_test.reset_index(drop = True, inplace = True)

#%%-----------------------------------------------------------------------

# Scaling the data, we need to fit and transform train data and just transfrom test data to avoid data leakage
label_enc = StandardScaler()

X_train_scaled_vals = label_enc.fit_transform(X_train[continuous_cols].values)

X_train_scaled = pd.DataFrame(data = X_train_scaled_vals, columns = X_train[continuous_cols].columns)

X_test_scaled_vals = label_enc.transform(X_test[continuous_cols].values)
X_test_scaled = pd.DataFrame(data = X_test_scaled_vals, columns = X_test[continuous_cols].columns)

X_train_cat = X_train[list(set(X_train.columns) - set(continuous_cols))]
X_test_cat = X_test[list(set(X_train.columns) - set(continuous_cols))]

#%%-----------------------------------------------------------------------

X_train_final = pd.concat([X_train_scaled, X_train_cat], axis = 1)
X_test_final = pd.concat([X_test_scaled, X_test_cat], axis = 1)

#%%-----------------------------------------------------------------------

print("Before balancing the data: ", Counter(y_train))

#%%-----------------------------------------------------------------------

# creating smotes object
rs = SMOTE()
# Applying smote to train data
X_train_smote, y_train_smote = rs.fit_resample(X_train_final, y_train)


#%%-----------------------------------------------------------------------

print("After balancing the data: ", Counter(y_train_smote))

#%%-----------------------------------------------------------------------
X_train = X_train_smote
X_test = X_test_final

y_train = y_train_smote
y_test = y_test
#%%-----------------------------------------------------------------------
#Now the data is prepared and we can move to training the model.
#%%-------------------------------------------------------------------#%%---------------------------------------------------------------------------
## Modelling2

def create_model(clf, X_train,X_test, y_train, y_test, decision_tree =False ):
        
    # fit the model
    clf.fit(X_train, y_train)
    
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    # Calculating Roc score
    roc_train = (roc_auc_score(y_train, clf.predict_proba(X_train.values)[:, 1]))    
    roc_test = (roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))    

    # Calculating fscore, precision & recall 
    f1_train = (f1_score(y_train, y_train_pred))
    f1_test = (f1_score(y_test, y_test_pred))

    precision_train = (precision_score(y_train, y_train_pred))
    precision_test=(precision_score(y_test, y_test_pred))

    recall_train = (recall_score(y_train, y_train_pred))        
    recall_test= (recall_score(y_test, y_test_pred))
    print()
    print("+","="*100,"+")    
    print(" F score on test set is:", f1_test )
    print("-"*50)
    print(" Precision on train set is:", precision_train)
    print(" Precision on test set is:", precision_test)
    print("-"*50)
    print(" Recall on train set is:",  recall_train)
    print(" Recall on test set is:",  recall_test)
    print("-"*50)
    print(' Train ROC is:',roc_train) 
    print(' Test ROC is:', roc_test )
    print("+","="*100,"+")  

 
    # plot no skill roc curve
    pyplot.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    # calculate roc curve for model
    fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
    # plot model roc curve
    pyplot.plot(fpr, tpr, marker='.', label='Dummy Model')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()      
    
    no_skill = len(y_test[y_test==1]) / len(y_test)
    # plot the no skill precision-recall curve
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    # calculate model precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, clf.predict_proba(X_test)[:, 1])
    # plot the model precision-recall curve
    pyplot.plot(recall, precision, marker='.', label='Model Performance')
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()
    
    if decision_tree:
        fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (50,40))

        tree.plot_tree(clf, fontsize=20, 
                       feature_names = X_train.columns, 
                       class_names = list(map(str, list(y.unique()))),
                       filled = True, max_depth = 5);
        plt.show()

    print("Feature Interpretation: ")
    
    return eli5.show_weights(clf, feature_names= list(X_train.columns))


#%%-----------------------------------------------------------------------

## Decision Tree

create_model(DecisionTreeClassifier(random_state = 0), X_train_smote, X_test_final, y_train_smote, y_test, True )

#%%-----------------------------------------------------------------------

## XGBOOST

create_model(XGBClassifier(random_state = 0), X_train_smote, X_test_final, y_train_smote, y_test, False )

#%%-----------------------------------------------------------------------
#%%-----------------------------------------------------------------------

##Observations:

#%%-----------------------------------------------------------------------
## Conclusion:
print("+","="*100
print('\033[1m' + f"\t\t\t\The train and test ROC looks pretty good. The score would be low even though we have created synthetic data using SMOTE but still the event rate is pretty bad.#Age and smoking status contributes alot to the model you can see. We see that smoking seems to have the highest feature importance.
" + '\033[0m')
print("+","="*100,"+")
#%%-----------------------------------------------------------------------



























## References: Stroke Prediction Dataset: Data file Retrieved from https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
