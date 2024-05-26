#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pandaX


# In[2]:


Ze_Test = pandaX.read_csv(r"C:\Users\20F20753\Downloads\test.csv")
Ze_Train = pandaX.read_csv(r"C:\Users\20F20753\Downloads\train.csv")


# In[3]:


Ze_Train = Ze_Train.drop(Ze_Train.iloc[:,[0, 1]], axis = 1)


# In[4]:


Ze_Train.info()


# In[5]:


Ze_Test = Ze_Test.drop(Ze_Test.iloc[:,[0, 1]], axis = 1)
Ze_Test.info()


# In[6]:


Ze_Train.columns = [c.replace(' ', '_') for c in Ze_Train.columns]
Ze_Test.columns = [c.replace(' ', '_') for c in Ze_Test.columns]


# In[7]:


Ze_Train['satisfaction'].replace({'neutral or dissatisfied': 0, 'satisfied': 1},inplace = True)
Ze_Test['satisfaction'].replace({'neutral or dissatisfied': 0, 'satisfied': 1},inplace = True)


# In[9]:


Ze_Total = Ze_Train.isnull().sum().sort_values(ascending=False)
percentage = (Ze_Train.isnull().sum()/Ze_Train.isnull().count()).sort_values(ascending=False)
inBalance = pandaX.concat([Ze_Total, percentage], axis=1, keys=['Total', 'Percent'])
inBalance.head()


# In[10]:


Ze_Train['Arrival_Delay_in_Minutes'] = Ze_Train['Arrival_Delay_in_Minutes'].fillna(Ze_Train['Arrival_Delay_in_Minutes'].mean())
Ze_Test['Arrival_Delay_in_Minutes'] = Ze_Test['Arrival_Delay_in_Minutes'].fillna(Ze_Test['Arrival_Delay_in_Minutes'].mean())


# In[11]:


Ze_Train.select_dtypes(include=['object']).columns


# In[14]:


Ze_Train['Gender'] = Ze_Train['Gender'].fillna(Ze_Train['Gender'].mode()[0])
Ze_Train['Customer_Type'] = Ze_Train['Customer_Type'].fillna(Ze_Train['Customer_Type'].mode()[0])
Ze_Train['Type_of_Travel'] = Ze_Train['Type_of_Travel'].fillna(Ze_Train['Type_of_Travel'].mode()[0])
Ze_Train['Class'] = Ze_Train['Class'].fillna(Ze_Train['Class'].mode()[0])


# In[15]:


Ze_Test['Gender'] = Ze_Test['Gender'].fillna(Ze_Test['Gender'].mode()[0])
Ze_Test['Customer_Type'] = Ze_Test['Customer_Type'].fillna(Ze_Test['Customer_Type'].mode()[0])
Ze_Test['Type_of_Travel'] = Ze_Test['Type_of_Travel'].fillna(Ze_Test['Type_of_Travel'].mode()[0])
Ze_Test['Class'] = Ze_Test['Class'].fillna(Ze_Test['Class'].mode()[0])


# In[17]:


from sklearn.preprocessing import LabelEncoder
lenCode = {}
for column in Ze_Train.select_dtypes(include=['object']).columns:
    lenCode[column] = LabelEncoder()
    Ze_Train[column] = lenCode[column].fit_transform(Ze_Train[column])


# In[18]:


lencoders_t = {}
for col in Ze_Test.select_dtypes(include=['object']).columns:
    lencoders_t[col] = LabelEncoder()
    Ze_Test[col] = lencoders_t[col].fit_transform(Ze_Test[col])


# In[19]:


Q1 = Ze_Train.quantile(0.25)
Q3 = Ze_Train.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[20]:


Ze_Train = Ze_Train[~((Ze_Train < (Q1 - 1.5 * IQR)) |(Ze_Train > (Q3 + 1.5 * IQR))).any(axis=1)]
Ze_Train.shape


# In[21]:


features = ['Type_of_Travel','Inflight_wifi_service','Online_boarding','Seat_comfort','Flight_Distance',
            'Inflight_entertainment','On-board_service','Leg_room_service','Cleanliness','Checkin_service', 
            'Inflight_service', 'Baggage_handling']
target = ['satisfaction']

# Split into test and train
x_train = Ze_Train[features]
y_train = Ze_Train[target].to_numpy()
x_test = Ze_Test[features]
y_test = Ze_Test[target].to_numpy()

# Normalize Features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


# In[24]:


import time
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, plot_confusion_matrix, plot_roc_curve
from matplotlib import pyplot as plt 
def run_model(model, x_train, y_train, x_test, y_test, verbose=True):
    t0=time.time()
    if verbose == False:
        model.fit(x_train,y_train.ravel(), verbose=0)
    else:
        model.fit(x_train,y_train.ravel())
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred) 
    time_taken = time.time()-t0
    print("Accuracy = {}".format(accuracy))
    print("ROC Area under Curve = {}".format(roc_auc))
    print("Time taken = {}".format(time_taken))
    print(classification_report(y_test,y_pred,digits=5))
    plot_confusion_matrix(model, x_test, y_test,cmap=plt.cm.pink, normalize = 'all')
    plot_roc_curve(model, x_test, y_test)                     
    
    return model, accuracy, roc_auc, time_taken


# In[23]:


from sklearn.neighbors import KNeighborsClassifier

params_kn = {'n_neighbors':10, 'algorithm': 'kd_tree', 'n_jobs':4}

model_kn = KNeighborsClassifier(**params_kn)
model_kn, accuracy_kn, roc_auc_kn, tt_kn = run_model(model_kn, x_train, y_train, x_test, y_test)


# In[25]:


from sklearn.tree import DecisionTreeClassifier
params_dt = {'max_depth': 12,    
             'max_features': "sqrt"}

model_dt = DecisionTreeClassifier(**params_dt)
model_dt, accuracy_dt, roc_auc_dt, tt_dt = run_model(model_dt, x_train, y_train, x_test, y_test)


# In[29]:


get_ipython().system('pip install graphviz')
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz

features_n = ['Type_of_Travel', 'Inflight_wifi_service', 'Online_boarding', 'Seat_comfort']
x_train_n = scaler.fit_transform(Ze_Train[features_n])
data = export_graphviz(DecisionTreeClassifier(max_depth=12).fit(x_train_n, y_train), out_file=None, 
                       feature_names = features_n,
                       class_names = ['Dissatisfied (0)', 'Satisfied (1)'], 
                       filled = True, rounded = True, special_characters = True)

graph = graphviz.Source(data)
graph


# In[ ]:




