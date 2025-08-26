# import necessary libraries and modules
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, matthews_corrcoef
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from autogluon.tabular import TabularDataset, TabularPredictor


######################################### Model 1 ########################################

#Loading the training and test data
df = pd.read_csv("train.csv") 
test_labels = pd.read_csv("test.csv")

train_sq = df['Sequence'].to_list()

"""# Converting to Fasta"""

fasta_file = "test.fasta"
with open(fasta_file,"w") as file:
    for i,seq in enumerate(train_sq):
        header = f">seq_{i+1}\n"
        file.write(header)
        file.write(seq+"\n")

# Extracting labels and from the traning data
tr_label = df['Label']

#load pre-processed features for traning from a csv file (physico - chemical property)
tr_aac = pd.read_csv("train_pcp.csv") #contains features

#Defining the feature matrix (X) and labels (Y) 
X = tr_aac

y = tr_label

#Defining a dictionary of classifiers to be used for traning  
classifiers = {
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(),
    'MLP': MLPClassifier(max_iter=1000),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'XGBoost': XGBClassifier(eval_metric='logloss')
}

#Use of cross-validation fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # 5-fold cross validation
metrics = {} #to store the performance metrices for each classifer

"""# Training Loop"""
#Loop through each classifer and train
for clf_name, clf in classifiers.items():
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    roc_auc_scores = []
    mcc_scores = []  # To store MCC scores

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

     # Convert -1 labels to 0 for compatibility
    y_binary = y.replace(-1, 0)

    # for performing 5-fold cross validation
    for train_idx, test_idx in skf.split(X_scaled, y_binary):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_binary.iloc[train_idx], y_binary.iloc[test_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]  # Probabilities for ROC-AUC

        # Calculating the metrics for the fold
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred, pos_label=1))
        recall_scores.append(recall_score(y_test, y_pred, pos_label=1))
        f1_scores.append(f1_score(y_test, y_pred, pos_label=1))
        roc_auc_scores.append(roc_auc_score(y_test, y_prob))
        mcc_scores.append(matthews_corrcoef(y_test, y_pred))

    # To store the mean and standard deviation of metrics across all folds
    metrics[clf_name] = {
        'Accuracy (Mean ± Std)': f"{np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}",
        'Precision (Mean ± Std)': f"{np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}",
        'Recall (Mean ± Std)': f"{np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}",
        'F1-Score (Mean ± Std)': f"{np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}",
        'ROC-AUC (Mean ± Std)': f"{np.mean(roc_auc_scores):.4f} ± {np.std(roc_auc_scores):.4f}",
        'MCC (Mean ± Std)': f"{np.mean(mcc_scores):.4f} ± {np.std(mcc_scores):.4f}"  # MCC added
    }

# Converting the metrics dictionary to a DataFrame and display
metrics_df = pd.DataFrame(metrics).T

test = pd.read_csv("test_pcp.csv")

test_scaled = scaler.transform(test)

test_scaled = pd.DataFrame(test_scaled,columns = test.columns)

probabilities = {}

# Iterating through classifiers
for clf_name, clf in classifiers.items():
    # Get probability predictions
    y_prob = clf.predict_proba(test_scaled)[:, 1]  # Probabilities for the positive class (1)
    probabilities[clf_name] = y_prob

# Converting it to a DataFrame for analysis
probabilities_df = pd.DataFrame(probabilities)

# Displaying the probabilities of the DataFrame
print(probabilities_df)

probabilities_df

"""## SVM Model"""

label_label_svm = probabilities_df['SVM']

label_svm = pd.DataFrame(label_label_svm)

label_svm.rename(columns={'SVM':'Label'},inplace=True)


#Concatenate the test labels with predicted labels for svm
result_svm = pd.concat([test_labels['ID'],label_svm],axis = 1)

#saving the svm file into .csv
result_svm.to_csv("PCP_Label_AG.csv",index = False)


######################################### Model 2 ###########################################
# Loading the traning dataset
train_aac = pd.read_csv("train_aac.csv") #Load the amino acid composition
train_dpc = pd.read_csv("train_dpc.csv") #Load the dipeptide composition
train_pcp = pd.read_csv("train_pcp.csv") #Load the physico-chemoical property
train_labels = pd.read_csv("train.csv") #Load the traning labels

#Extract the sequences from the training data
train_seq = train_labels['Sequence']

#Concatenate AAC, DPC and PCP 
train = pd.concat([train_aac,train_dpc,train_pcp,train_labels['Label']],axis=1)

"""Test Set"""

#Loading the test dataset
test_aac = pd.read_csv("test_aac.csv")
test_dpc = pd.read_csv("test_dpc.csv")
test_pcp = pd.read_csv("test_pcp.csv")


test_seq = test_labels['Sequence']

#Concatenate the AAC , DPC and PCP features into a single test dataset
test = pd.concat([test_aac,test_dpc,test_pcp],axis=1)

"""Model"""

#import AutoGluon tabular components
from autogluon.tabular import TabularDataset, TabularPredictor

#Converting the traning dataset into a AutoGluon Tabular format
train_data = TabularDataset(train)
train_data.head()

#shuffling the traning data and set a fixed random seed
train_data = train_data.sample(frac=1,random_state=7)

#defining the test size for splitting the traning data
test_size = 0.2
num_test = test_size*len(train_data)

shuffled_df = train_data.sample(frac=1,random_state=7)

#split the data into traning and test sets
train_df = shuffled_df[int(num_test):]
test_df = shuffled_df[:int(num_test)]

#defining the label and path for Autogluon
label = 'Label'
save_path = 'AAC_AutoG_BQ5'
metric = 'roc_auc'

#initialize the TabularPredictor for classification tasks
predictor = TabularPredictor(label=label, path=save_path,eval_metric=metric)

#Train the model using AutoGluon
predictor.fit(train_data,presets='best_quality',num_bag_folds=5,num_stack_levels=1,num_bag_sets=1)

#Get a summary of the model
results = predictor.fit_summary()


test = TabularDataset(test)

#Getting the predicted probabilites for each class
test_pred = predictor.predict(test)

#Get the predicted probabilites of each class
test_pred_prob = predictor.predict_proba(test)

#Convert the predictions and probabilites to dataframe for further processsing
test_pred = pd.DataFrame(test_pred)

#Get the probabilites for positive class
test_pred_prob = pd.DataFrame(test_pred_prob.iloc[:,-1])

#Concatenate the test sequence IDs with the predicted probabilites
prob_predictions = pd.concat([test_labels['ID'],test_pred_prob],axis=1)

prob_predictions.rename(columns={1:'Label'},inplace=True)

#Save the file to a CSV file
prob_predictions.to_csv("AAC,DPC,PCP_Label_AG.csv",index=False)


############################################ Model 3 ###########################################
train_aac = pd.read_csv("train_aac.csv")  #Load the amino acid composition
train_dpc = pd.read_csv("train_dpc.csv")  #Load the dipeptide composition
train_pcp = pd.read_csv("train_pcp.csv")  #Load the physico-chemoical property
train_rri = pd.read_csv("train_rri.csv")  # Load the residue repeat information
train_labels = pd.read_csv("train.csv")   #Load the traning labels

#Extract the sequences from the training data
train_seq = train_labels['Sequence']

#Concatenate AAC, DPC, PCP and RRI
train = pd.concat([train_aac,train_dpc,train_pcp,train_rri,train_labels['Label']],axis=1)

"""Test Set"""

#Loading the test dataset
test_aac = pd.read_csv("test_aac.csv")
test_dpc = pd.read_csv("test_dpc.csv")
test_pcp = pd.read_csv("test_pcp.csv")
test_rri = pd.read_csv("test_rri.csv")
test_labels = pd.read_csv("test.csv")

test_seq = test_labels['Sequence']

#Concatenate the AAC , DPC and PCP features into a single test dataset
test = pd.concat([test_aac,test_dpc,test_pcp],axis=1)

"""Model"""

#import AutoGluon tabular components
from autogluon.tabular import TabularDataset, TabularPredictor

#Converting the traning dataset into a AutoGluon Tabular format
train_data = TabularDataset(train)
train_data.head()

#shuffling the traning data and set a fixed random seed
train_data = train_data.sample(frac=1,random_state=7)

#defining the test size for splitting the traning data
test_size = 0.2
num_test = test_size*len(train_data)

shuffled_df = train_data.sample(frac=1,random_state=7)

#split the data into traning and test sets
train_df = shuffled_df[int(num_test):]
test_df = shuffled_df[:int(num_test)]

#defining the label and path for Autogluon
label = 'Label'
save_path = 'AAC_AutoG_BQ6'
metric = 'roc_auc'
predictor = TabularPredictor(label=label, path=save_path,eval_metric=metric)

#defining the label and path for Autogluon
predictor.fit(train_data,presets='best_quality',num_bag_folds=5,num_stack_levels=1,num_bag_sets=1)

#Get a summary of the model
results = predictor.fit_summary()

test = TabularDataset(test)

#Getting the predicted probabilites for each class
test_pred = predictor.predict(test)

#Get the predicted probabilites of each class
test_pred_prob = predictor.predict_proba(test)

test_pred = pd.DataFrame(test_pred)

#Get the probabilites for positive class
test_pred_prob = pd.DataFrame(test_pred_prob.iloc[:,-1])

#Concatenate the test sequence IDs with the predicted probabilites
prob_predictions = pd.concat([test_labels['ID'],test_pred_prob],axis=1)

prob_predictions.rename(columns={1:'Label'},inplace=True)

#Save the file to a CSV file
prob_predictions.to_csv("AAC,DPC,PCP,RRI_Label_AG.csv",index=False)