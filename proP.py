import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import CategoricalImputer
import numpy as np
import pandas as pd
from pandas import MultiIndex
from sklearn.model_selection import train_test_split

class pre:
    def preprocess_for_predict(data):
        """Takes Path as arugment and returns data frame  """
        
        #drop irrelvent data for prediction
        col_drop = ['policy_number','policy_bind_date','policy_state','insured_zip','incident_location','incident_date','incident_state','incident_city','insured_hobbies','auto_make','auto_model','auto_year','total_claim_amount','age']
        data.drop(columns=col_drop,inplace=True)
        data = data.replace('?',np.nan)

        #CategoricalImputer
        imputer = CategoricalImputer()
        data['collision_type']              = imputer.fit_transform(data['collision_type'])
        data['property_damage']             = imputer.fit_transform(data['property_damage'])
        data['police_report_available']     = imputer.fit_transform(data['police_report_available'])
        cat_data = data.select_dtypes(include=['object']).copy()
        num_data = data.select_dtypes(include=['int64','float64']).copy()

        #mapping few features 
        cat_data['policy_csl']                = cat_data['policy_csl'].map({'100/300' : 1, '250/500' : 2.5 ,'500/1000':5})
        cat_data['insured_education_level']   = cat_data['insured_education_level'].map({'JD' : 1, 'High School' : 2,'College':3,'Masters':4,'Associate':5,'MD':6,'PhD':7})
        cat_data['incident_severity']         = cat_data['incident_severity'].map({'Trivial Damage' : 1, 'Minor Damage' : 2,'Major Damage':3,'Total Loss':4})
        cat_data['insured_sex']               = cat_data['insured_sex'].map({'FEMALE' : 0, 'MALE' : 1})
        cat_data['property_damage']           = cat_data['property_damage'].map({'NO' : 0, 'YES' : 1})
        cat_data['police_report_available']   = cat_data['police_report_available'].map({'NO' : 0, 'YES' : 1})
        
        
        #using get dummies with first drop = True

        for col_title in cat_data.drop(columns=['policy_csl','insured_education_level','incident_severity','insured_sex','property_damage','police_report_available']).columns:
            cat_data = pd.get_dummies(cat_data, columns=[col_title], prefix = [col_title],drop_first=True) # to avoid dummy trap drop first column
        
        #joint cat_data and num_data
        scaler=StandardScaler()
        num_data.head()
        num_data1=scaler.fit_transform(num_data)
        num_data1= pd.DataFrame(data=num_data, columns=num_data.columns)
        training_feed = pd.concat([num_data1,cat_data],axis=1) # axies =1 means side by side concat

        print('PreProcessing for Prediction Done Succesfully')
        return training_feed
    
    def preprocess_for_train(path):
        """Takes Path as arugment and returns data frame  """
        data = pd.read_csv(path)
        #drop irrelvent data for training
        col_drop = ['policy_number','policy_bind_date','policy_state','insured_zip','incident_location','incident_date','incident_state','incident_city','insured_hobbies','auto_make','auto_model','auto_year','total_claim_amount']
        data.drop(columns=col_drop,inplace=True)
        data = data.replace('?',np.nan)

        #CategoricalImputer
        imputer = CategoricalImputer()
        data['collision_type']              = imputer.fit_transform(data['collision_type'])
        data['property_damage']             = imputer.fit_transform(data['property_damage'])
        data['police_report_available']     = imputer.fit_transform(data['police_report_available'])
        cat_data = data.select_dtypes(include=['object']).copy()
        num_data = data.select_dtypes(include=['int64','float64']).copy()

        #mapping few features 
        cat_data['policy_csl']                = cat_data['policy_csl'].map({'100/300' : 1, '250/500' : 2.5 ,'500/1000':5})
        cat_data['insured_education_level']   = cat_data['insured_education_level'].map({'JD' : 1, 'High School' : 2,'College':3,'Masters':4,'Associate':5,'MD':6,'PhD':7})
        cat_data['incident_severity']         = cat_data['incident_severity'].map({'Trivial Damage' : 1, 'Minor Damage' : 2,'Major Damage':3,'Total Loss':4})
        cat_data['insured_sex']               = cat_data['insured_sex'].map({'FEMALE' : 0, 'MALE' : 1})
        cat_data['property_damage']           = cat_data['property_damage'].map({'NO' : 0, 'YES' : 1})
        cat_data['police_report_available']   = cat_data['police_report_available'].map({'NO' : 0, 'YES' : 1})
        cat_data['fraud_reported']            = cat_data['fraud_reported'].map({'N' : 0, 'Y' : 1})
        
        #using get dummies with first drop = True

        for col_title in cat_data.drop(columns=['policy_csl','insured_education_level','incident_severity','insured_sex','property_damage','police_report_available','fraud_reported']).columns:
            cat_data = pd.get_dummies(cat_data, columns=[col_title], prefix = [col_title],drop_first=True)
        
        #joint cat_data and num_data
        scaler=StandardScaler()
        num_data.head()
        num_data1=scaler.fit_transform(num_data)
        num_data1= pd.DataFrame(data=num_data, columns=num_data.columns)
        training_feed = pd.concat([num_data1,cat_data],axis=1)

        print('PreProcessing Done Succesfully')
        return training_feed

class training:
    def get_train_sets(feed):
        x = feed.drop('fraud_reported',axis =1)
        y = feed['fraud_reported']
        train_x,test_x,train_y,test_y=train_test_split(x,y, random_state=355)

        return train_x,test_x,train_y,test_y
        


