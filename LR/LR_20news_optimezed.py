# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 01:50:57 2021

@author: alexa
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import logging
from sklearn.datasets import fetch_20newsgroups
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import metrics

# FUNCTION: extract_features()
# specified word counting limits: (`min_df`, `max_df`)
#
#
def extract_features(training_data,testing_data,type="binary"):
    """Extract features using different methods"""

    logging.info("Extracting features and creating vocabulary...")
    
    if "binary" in type: # BINARY FEATURE REPRESENTATION
        cv= CountVectorizer(binary=True, max_df=0.95)
  
    elif "counts" in type: # COUNT BASED FEATURE REPRESENTATION
        cv= CountVectorizer(binary=False, max_df=0.95)
    
    else: # TF-IDF BASED FEATURE REPRESENTATION
        cv=TfidfVectorizer(use_idf=True, max_df=0.95)
        
    cv.fit_transform(training_data.data) #creates a vocabulary based on the training set
    
    #cv. transform:
    #takes in any text (test or unseen texts) and transforms it according to the vocabulary of the training set, limiting the words by the specified count restrictions (`min_df`, `max_df`) and applying necessary stop words if specified
    #returns a term-document matrix where each column in the matrix represents a word in the vocabulary while each row represents the documents in the dataset. The values could either be binary or counts
    train_feature_set=cv.transform(training_data.data)
    test_feature_set=cv.transform(testing_data.data)
    
    return train_feature_set,test_feature_set,cv

def get_top_k_predictions(model,X_test,k): 
    # get probabilities instead of predicted labels, since we want to collect top 3
    probs = model.predict_proba(X_test)
    
    # GET TOP K PREDICTIONS BY PROB - note these are just index
    best_n = np.argsort(probs, axis=1)[:,-k:]
        
    # GET CATEGORY OF PREDICTIONS
    preds=[[model.classes_[predicted_cat] for predicted_cat in prediction] for prediction in best_n]
        
    # REVERSE CATEGORIES - DESCENDING ORDER OF IMPORTANCE
    preds=[ item[::-1] for item in preds]
        
    return preds

def _reciprocal_rank(true_labels: list, machine_preds: list):
    """Compute the reciprocal rank at cutoff k"""
    
    # add index to list only if machine predicted label exists in true labels
    tp_pos_list = [(idx + 1) for idx, r in enumerate(machine_preds) if r in true_labels]

    rr = 0
    if len(tp_pos_list) > 0:
        # for RR we need position of first correct item
        first_pos_list = tp_pos_list[0]
        
        # rr = 1/rank
        rr = 1 / float(first_pos_list)

    return rr

def compute_mrr_at_k(items:list):
    """Compute the MRR (average RR) at cutoff k"""
    rr_total = 0
    
    for item in items:   
        rr_at_k = _reciprocal_rank(item[0],item[1])
        rr_total = rr_total + rr_at_k
        mrr = rr_total / 1/float(len(items))

    return mrr

def collect_preds(Y_test,Y_preds):
    """Collect all predictions and ground truth"""
    
    pred_gold_list=[[[Y_test[idx]],pred] for idx,pred in enumerate(Y_preds)]
    return pred_gold_list

def compute_accuracy(eval_items:list):
    correct=0
    total=0
    
    for item in eval_items:
        true_pred=item[0]
        machine_pred=set(item[1])
        
        for cat in true_pred:
            if cat in machine_pred:
                correct+=1
                break
    
    
    accuracy=correct/float(len(eval_items))
    return accuracy
             
def train_model(training_data,testing_data,solver='liblinear',penalty='l2', field="text_desc",feature_rep="binary",top_k=3):
    
    logging.info("Starting model training...")
    print("feat_rep="+feature_rep+"  top_k="+ str(top_k)+"\n")

    # GET LABELS
    Y_train=training_data.target
    Y_test=testing_data.target
         
    # GET FEATURES
    X_train,X_test,feature_transformer=extract_features(training_data,testing_data,type=feature_rep)

    # INIT LOGISTIC REGRESSION CLASSIFIER
    logging.info("Training a Logistic Regression Model...")
    if penalty=='elasticnet':
        l1_ratio=0.5
    else:
        l1_ratio=None
                
    scikit_log_reg = LogisticRegression(verbose=1, solver=solver,l1_ratio=l1_ratio,random_state=0, C=5, penalty=penalty,max_iter=1000)
    model=scikit_log_reg.fit(X_train,Y_train)

    # GET TOP K PREDICTIONS
    preds=get_top_k_predictions(model,X_test,top_k)
    
    # GET PREDICTED VALUES AND GROUND TRUTH INTO A LIST OF LISTS - for ease of evaluation
    eval_items=collect_preds(Y_test,preds)
    
    # GET EVALUATION NUMBERS ON TEST SET -- HOW DID WE DO?
    logging.info("Starting evaluation...")
    #accuracy=accuracy_score(testing_data.target, preds) #can only be used when top_k=1
    accuracy=compute_accuracy(eval_items)
    mrr_at_k=compute_mrr_at_k(eval_items)
    
    logging.info("Done training and evaluation.")
    
    return model,feature_transformer,accuracy,mrr_at_k,preds

def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Load the dataset
    data = fetch_20newsgroups()
    # Get the text categories
    text_categories = data.target_names
    # define the training set
    training_data = fetch_20newsgroups(subset="train", categories=text_categories)
    # define the test set
    testing_data = fetch_20newsgroups(subset="test", categories=text_categories)

    
    #feature_reps=['binary','counts','tfidf'] #these are the 3 types of metrics that will be tested
    feature_reps=['tfidf']
    top_ks=[1,3,5]
    results=[]
    solvers=['liblinear']
    penalties=['l2']
    
    #model hyperparameters:   
    #solvers=['liblinear','newton-cg','sag','saga']
    #penalties=['l2','l1','elasticnet']
    
    for feature_rep in feature_reps:
        for top_k in top_ks:
            for solver in solvers:
                for penalty in penalties:
                    if penalty=='l1' and solver!='liblinear':
                        continue
                    if penalty=='elasticnet' and solver!='saga':
                        continue
                    model,transformer,acc,mrr_at_k,predicted_categories=train_model(training_data,testing_data,solver=solver, penalty=penalty,feature_rep=feature_rep,top_k=top_k)
                    #results.append([feature_rep,top_k,acc,mrr_at_k])#,solver,penalty])
                    #results.append([feature_rep,acc])
                    results.append([top_k,acc,mrr_at_k])
                    
                    if top_k==1:
                        print(metrics.classification_report(testing_data.target, predicted_categories,target_names=text_categories)) 
                        
                        # plot the confusion matrix
                        mat = confusion_matrix(testing_data.target, predicted_categories)
                        plt.figure(num=None, figsize=(10, 10), dpi=80)
                        sns.heatmap(mat.T, square = True, annot=True, fmt = "d", xticklabels=training_data.target_names,yticklabels=training_data.target_names)
            
                        plt.xlabel("true labels")
                        plt.ylabel("predicted label")
            
                        plt.savefig(feature_rep+'_LR'+solver+penalty+'.png', bbox_inches='tight')
                        plt.show()
            
                        print("The accuracy is {}".format(accuracy_score(testing_data.target, predicted_categories)))
                
    #df_results=pd.DataFrame(results,columns=['feature_representation','top_k','accuracy','mrr_at_k'])#,'solver','penalty'])
    #df_results=pd.DataFrame(results,columns=['feature_representation','accuracy'])
    df_results=pd.DataFrame(results,columns=['top_k','accuracy','mrr_at_k'])
    df_results.sort_values(by=['accuracy'],ascending=False)
    print(df_results)
    #model,transformer,accuracy,mrr_at_k=train_model(training_data,testing_data,feature_rep="binary",top_k=3)
    #print("\nAccuracy={0}; MRR={1}".format(accuracy,mrr_at_k))
    

if __name__ == "__main__":
    main()