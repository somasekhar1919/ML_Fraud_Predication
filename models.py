import xgboost
from  proP import pre, training   #proP.py file
#models library
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import accuracy_score
import pickle
import proP

feed = 'InputFile.csv'
feed = proP.pre.preprocess_for_train(feed)

def save_model(feed):
    
    modelName = ['SVC','xgboost','RFC']
    scores = []
    for i in range(3):

        modelname = modelName[i]
        train_x,test_x,train_y,test_y = training.get_train_sets(feed)            # this tuple --> dataset = ("train_x","test_x","train_y","test_y")
        m0 = SVC()
        m1 = RandomForestClassifier()
        m2 = XGBClassifier(use_label_encoder=False)
        fun = (m0, m1,m2)
        model = fun[i]
        model.fit(train_x, train_y)
        filename ='saved_models/'+ modelname +'.sav'
        pickle.dump(model, open(filename, 'wb'))
        y = model.predict(test_x)
        score = accuracy_score(y,test_y)
        scores.append(score)
        
        print(modelname+"="+str(score))
        print('Training Done successfully')
    return modelName,scores


