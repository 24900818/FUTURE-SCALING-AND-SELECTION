# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

       import pandas as pd
       import numpy as np
       import seaborn as sns
       from sklearn.model_selection import train_test_split
       from sklearn.neighbors import KNeighborsClassifier
       from sklearn.metrics import accuracy_score, confusion_matrix
       data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
       data

![image](https://github.com/user-attachments/assets/8cbbf7b9-c7a5-46b3-98d0-7be116fb2000)

        data.isnull().sum()

![image](https://github.com/user-attachments/assets/0e34c036-7e04-41bd-839b-e4295cb9994f)

        missing=data[data.isnull().any(axis=1)]
        missing

![image](https://github.com/user-attachments/assets/25c33c85-3546-40be-a2cd-ec8536a5c0e8)

        data2=data.dropna(axis=0)
        data2

![image](https://github.com/user-attachments/assets/e2700554-ad08-43dd-a754-4de2ade74eef)

        sal=data["SalStat"]
        data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
        print(data2['SalStat'])

![image](https://github.com/user-attachments/assets/e40bae20-0cdc-47c5-af6c-0dd7668dea2e)

        sal2=data2['SalStat']
        dfs=pd.concat([sal,sal2],axis=1)
        dfs

![image](https://github.com/user-attachments/assets/1e187afc-7dfe-49bb-92a5-d715194a8377)

       data2

![image](https://github.com/user-attachments/assets/8ca63a24-a075-4d05-bfbf-9c7a569a55e0)

       new_data=pd.get_dummies(data2, drop_first=True)
       new_data

![image](https://github.com/user-attachments/assets/8689ecb2-7209-46dc-86d3-4c379e8f5784)

       columns_list=list(new_data.columns)
       print(columns_list)

![image](https://github.com/user-attachments/assets/207f2392-2cd0-4946-836b-a626dad615cb)

        features=list(set(columns_list)-set(['SalStat']))
        print(features)

![image](https://github.com/user-attachments/assets/054aa639-a552-4fad-a5a0-fe49a2b7c790)

        y=new_data['SalStat'].values
        print(y)

![image](https://github.com/user-attachments/assets/86afad6f-2581-4003-9291-09ccfcedd905)

       x=new_data[features].values
       print(x)
       
![image](https://github.com/user-attachments/assets/ee61f498-142d-4d61-a384-6f57d1012dd2)

       train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
       KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
       KNN_classifier.fit(train_x,train_y)

![image](https://github.com/user-attachments/assets/0f54a1fe-ff1e-4277-8efc-4c21a7182f8a)
     
        prediction=KNN_classifier.predict(test_x)
        confusionMatrix=confusion_matrix(test_y, prediction)
        print(confusionMatrix)

![image](https://github.com/user-attachments/assets/da20bbcf-27a9-4fb4-8ae5-5e110113fed0)

        accuracy_score=accuracy_score(test_y,prediction)
        print(accuracy_score)

![image](https://github.com/user-attachments/assets/bb88f06a-f295-4c53-be58-9bf01407fc01)

       print("Misclassified Samples : %d" % (test_y !=prediction).sum())

![image](https://github.com/user-attachments/assets/7b1917ea-5db5-4b9b-886a-79df6a479366)

       data.shape

 ![image](https://github.com/user-attachments/assets/bcf75849-b6cc-4bb8-b9cc-a7ef511329e6)

        import pandas as pd
        from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
        data={
              'Feature1': [1,2,3,4,5],
              'Feature2': ['A','B','C','A','B'],
              'Feature3': [0,1,1,0,1],
              'Target'  : [0,1,1,0,1]
       }

       df=pd.DataFrame(data)
       x=df[['Feature1','Feature3']]
       y=df[['Target']]
       selector=SelectKBest(score_func=mutual_info_classif,k=1)
       x_new=selector.fit_transform(x,y)
       selected_feature_indices=selector.get_support(indices=True)
       selected_features=x.columns[selected_feature_indices]
       print("Selected Features:")
       print(selected_features)

![image](https://github.com/user-attachments/assets/1110cbab-4a29-4fde-8ad8-861d8c7feda9)

       import pandas as pd
       import numpy as np
       from scipy.stats import chi2_contingency
       import seaborn as sns
       tips=sns.load_dataset('tips')
       tips.head()      

![image](https://github.com/user-attachments/assets/e864181f-639b-479b-adf4-b08068675f7e)

        tips.time.unique()



        contingency_table=pd.crosstab(tips['sex'],tips['time'])
        print(contingency_table)
        
![image](https://github.com/user-attachments/assets/a4182f97-9e3d-4664-b54b-09769bb3f586)

          chi2,p,_,_=chi2_contingency(contingency_table)
          print(f"Chi-Square Statistics: {chi2}")
          print(f"P-Value: {p}")

![image](https://github.com/user-attachments/assets/f67074cb-a0ef-4edf-976b-b6fc8851ada7)


# RESULT:
           Thus, Feature selection and Feature scaling has been used on thegiven dataset.
