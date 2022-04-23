import pickle

import pandas as pd

import proP 
#path = '/home/juniorcoder/Documents/ML project/InputFile copy.csv'
def predict(path):
    x_dummy = pd.read_csv('Dummy.csv')
    feed = pd.read_csv(path)
    new_feed = pd.concat([feed,x_dummy],axis=0) # stack the data 
    x = proP.pre.preprocess_for_predict(new_feed)
    x = x.iloc[:-14,:]       # removing dummy records
    new_feed.to_csv('concat.csv')
    modelname = ['SVC','xgboost','RFC']
    loaded_model = pickle.load(open('saved_models/'+ modelname[2]+'.sav', 'rb'))
    result = loaded_model.predict(x)
    report =[]
    for res in result:
        if res == 0:
            report.append('N')
        else:
            report.append('Y')
    feed['fraud report'] = report
    #col_drop = ['months_as_customer','age','policy_bind_date','policy_state','policy_csl','policy_deductable','policy_annual_premium','umbrella_limit','insured_zip','insured_sex','insured_education_level','insured_occupation','insured_hobbies','insured_relationship','capital-gains','capital-loss','incident_date','incident_type','collision_type','incident_severity','authorities_contacted','incident_state','incident_city','incident_location','incident_hour_of_the_day','number_of_vehicles_involved','property_damage','bodily_injuries','witnesses','police_report_available','total_claim_amount','injury_claim','property_claim','vehicle_claim','auto_make','auto_model','auto_year']
    #feed.drop(columns=col_drop,inplace=True)
    #feed = feed[['policy_number','fraud_reported']]
    feed.to_csv('upload_tasks/done/final.csv')
    return feed
