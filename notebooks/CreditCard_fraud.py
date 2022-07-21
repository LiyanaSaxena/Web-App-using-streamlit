# import the necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit 
import warnings 
warnings.filterwarnings("ignore")
import streamlit as st 

st.title('Credit Card Fraud Detection')

df=pd.read_csv('C:/Users/Admin/Downloads/creditcard.csv/creditcard.csv')

#Print Shape and description of the data 
if st.sidebar.checkbox('Show what the dataframe looks like'):
    st.write(df.head(100))
    st.write('Shape of the dataframe: ',df.shape)
    st.write('Data Description: \n',df.describe())

#Print valid and fraud transactions 
fraud = df[df.Class==1]
valid = df[df.Class==0]
outlier_percentage = (df.Class.value_counts()[1]/df.Class.value_counts()[0])*100

if st.sidebar.checkbox('Show fraud and valid transaction details'):
    st.write('Fraudulent transactions are: %.3f%%'%outlier_percentage)
    st.write('Fraud Cases: ',len(fraud))
    st.write('Valid Cases: ',len(valid))

#Under Sampling (Building a sample dataset with similar distribution of valid and fraud transactions)
#Number of fraud transactions are 492 
#To balance the dataset, we are taking 492 sample values among the total valid transactions
valid_sample = valid.sample(n=492)
#Creating a new dataset by joining valid_sample and fraud transactions
new_df=pd.concat([valid_sample,fraud],axis=0)

#Splithe data into features and labels
X=new_df.drop(['Class'], axis=1)
Y=new_df['Class']

#Split data into training and testing sets
from sklearn.model_selection import train_test_split
size = st.sidebar.slider('Test Set Size',min_value=0.2,max_value=0.4)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=size,random_state=2)

#Print Shape of train and test sets
if st.sidebar.checkbox('Show the shape of training and test set features and labels'):
    st.write('X_train: ',X_train)
    st.write('X_test: ',X_test)
    st.write('Y_train: ',Y_train)
    st.write('Y_test: ',Y_test)

#Import Classification models and metrics 
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import xgboost as xgb
import catboost as ctb

lr= LogisticRegression()
dtc = tree.DecisionTreeClassifier(random_state=0)
rf=RandomForestClassifier(random_state=0)
cls=svm.SVC(kernel='linear',probability=True)
xgb_cl=xgb.XGBClassifier()
cbc=ctb.CatBoostClassifier()

features=X_train.columns.tolist()

#Feature selection through feature importance
@st.cache
def feature_sort(model,X_train,Y_train):
    # fit the model
    model.fit(X_train, Y_train)
    # get importance
    imp = model.feature_importances_
    return imp
#Classifiers for feature importance
clf=['Decision Tree','Random Forest','Support Vector Machine','XG Boost','Cat Boost']
mod_feature = st.sidebar.selectbox('Which model for feature importance?', clf)
start_time = timeit.default_timer()
if mod_feature=='Decision Tree':
    model=dtc
    importance=feature_sort(model,X_train,Y_train)
elif mod_feature=='Random Forest':
    model=rf
    importance=feature_sort(model,X_train,Y_train)
elif mod_feature=='Support Vector Machine':
    model=cls
    importance=feature_sort(model,X_train,Y_train)
elif mod_feature=='XG Boost':
    model=xgb_cl
    importance=feature_sort(model,X_train,Y_train)
elif mod_feature=='Cat Boost':
    model=cbc
    importance=feature_sort(model,X_train,Y_train)
elapsed = timeit.default_timer() - start_time
st.write('Execution Time for feature selection: %.2f minutes'%(elapsed/60))

st.set_option('deprecation.showPyplotGlobalUse', False)

#Plot of feature importance
if st.sidebar.checkbox('Show plot of feature importance'):
    plt.bar([x for x in range(len(importance))], importance)
    plt.title('Feature Importance')
    plt.xlabel('Feature (Variable Number)')
    plt.ylabel('Importance')
    st.pyplot()

feature_imp=list(zip(features,importance))
feature_sort=sorted(feature_imp, key = lambda x: x[1])

n_top_features = st.sidebar.slider('Number of top features', min_value=5, max_value=20)

top_features=list(list(zip(*feature_sort[-n_top_features:]))[0])

if st.sidebar.checkbox('Show selected top features'):
    st.write('Top %d features in order of importance are: %s'%(n_top_features,top_features[::-1]))

X_train_sfs=X_train[top_features]
X_test_sfs=X_test[top_features]

X_train_sfs_scaled=X_train_sfs
X_test_sfs_scaled=X_test_sfs




#CREATING A HELPER FUNCTION TO EVALUATE EACH TRAINED MODEL 
def evaluate_model(model, X_test, Y_test):
    from sklearn import metrics

    # Predict Test Data 
    Y_pred = model.predict(X_test)

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(Y_test, Y_pred)
    prec = metrics.precision_score(Y_test, Y_pred)
    rec = metrics.recall_score(Y_test, Y_pred)
    f1 = metrics.f1_score(Y_test, Y_pred)
    kappa = metrics.cohen_kappa_score(Y_test, Y_pred)

    # Calculate area under curve (AUC)
    Y_pred_proba = model.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(Y_test, Y_pred_proba)
    auc = metrics.roc_auc_score(Y_test, Y_pred_proba)

    # Display confussion matrix
    cm = metrics.confusion_matrix(Y_test, Y_pred)

    'Accuracy: ', acc 
    'Confusion Matrix:\n', cm
    'Precision: ', prec
    'Recall: ', rec
    'F1 Score:: ', f1
    'Cohens Kappa Score: ', kappa, 
    'Area Under Curve: ', auc

#Run different classification models with rectifiers
if st.sidebar.checkbox('Run a credit card fraud detection model'):
  alg=['Logistic Regression','Decision Tree','Random Forest','Support Vector Machine','XG Boost','Cat Boost']
  classifier = st.sidebar.selectbox('Which algorithm?', alg)

  if classifier=='Logistic Regression':
        model=lr
        model.fit(X_train_sfs_scaled,Y_train)
        evaluate_model(model,X_test_sfs_scaled,Y_test)
        
  elif classifier=='Decision Tree':
        model=dtc
        model.fit(X_train_sfs_scaled,Y_train)
        evaluate_model(model,X_test_sfs_scaled,Y_test)
       
        
  elif classifier=='Random Forest':
        model=rf
        model.fit(X_train_sfs_scaled,Y_train)
        evaluate_model(model,X_test_sfs_scaled,Y_test)

  elif classifier=='Support Vector Machine':
        model=cls
        model.fit(X_train_sfs_scaled,Y_train)
        evaluate_model(model,X_test_sfs_scaled,Y_test)

  elif classifier=='XG Boost':
        model=xgb_cl
        model.fit(X_train_sfs_scaled,Y_train)
        evaluate_model(model,X_test_sfs_scaled,Y_test)

  elif classifier=='Cat Boost':
        model=cbc
        model.fit(X_train_sfs_scaled,Y_train)
        evaluate_model(model,X_test_sfs_scaled,Y_test)

# st.write('Accuracy:', evaluate_model['acc'])
# st.write('Precision:', evaluate_model['prec'])
# st.write('Recall:', evaluate_model['rec'])
# st.write('F1 Score:', evaluate_model['f1'])
# st.write('Cohens Kappa Score:', evaluate_model['kappa'])
# st.write('Area Under Curve:', evaluate_model['auc'])
# st.write('Confusion Matrix:\n', evaluate_model['cm'])