import flask as flask
from flask import Flask

import pandas as pd
import numpy as np

import sklearn
import joblib
from flask import Flask, jsonify, request
import json
from treeinterpreter import treeinterpreter as ti
from sklearn.neighbors import NearestNeighbors
import pickle
from lightgbm import LGBMClassifier
import shap

with open("lgbm_model.pickle", 'rb') as f:
    lgbm_model = pickle.load(f, encoding='latin-1')

#with open("lgbm_model.pickle") as f:
#    lgbm_model= pickle.load(f)
X_train = pd.read_pickle("X_train.pickle")
#X_train=X_train.sample(100)
#print(X_train.head(3))
#X_train.set_index("SK_ID_CURR", inplace=True)
y_train = pd.read_pickle("y_train.pickle")
#y_train=y_train.sample(100)
X_test_id = pd.read_pickle("X_test_id.pickle")
#X_test_id =X_test_id.sample(100)
# aggregated data of the train set for comparison to current applicant

#features_desc = pd.read_csv("feat_desc.csv", index_col=0)

# Set 'Name' column as the index
#X_test_id.set_index('SK_ID_CURR', inplace=True)
#X_test_id.fillna(data_for_display.mean(), inplace=True)
#print(X_test_id.head())
explainer = shap.Explainer(lgbm_model)
shap_values = explainer.shap_values(X_test_id)



sk_ids = list(X_test_id.index)

indices =np.arange(0,len(sk_ids))
dic_indices=dict(zip(sk_ids,indices))
app = Flask(__name__)


@app.route('/api/', methods=['GET'])
def home():
    return "<h1>API, model and data loaded</h1><p> This site"

# @app.route('/api/skids', methods=['GET'])
# def skids():
#     return "<h1> DER MOUSS</h1><p>"


@app.route('/api/sk_ids/', methods=['GET'])
# Test : http://127.0.0.1:5000/api/sk_ids/
def sk_ids():
    # Extract list of 'SK_ID_CURR' from the DataFrame
    sk_ids = list(X_test_id.index)

    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'data': sk_ids
     })


@app.route('/api/scoring/', methods=['GET'])
# Test : http://127.0.0.1:5000/api/scoring?SK_ID_CURR=137160
def scoring():
    # Parsing the http request to get arguments (applicant ID)
    SK_ID_CURR = int (request.args.get('SK_ID_CURR'))


    # Getting the data for the applicant (pd.DataFrame)
    applicant_data = X_test_id.loc[SK_ID_CURR:SK_ID_CURR]

    # Converting the pd.Series to dict
    applicant_score = 100*lgbm_model.predict_proba(applicant_data)[0][1]

    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'SK_ID_CURR': SK_ID_CURR,
        'score': applicant_score,
        'applicant_data' : applicant_data.to_json()
     })


# find 20 nearest neighbors among the training set
def get_df_neigh(sk_id_cust):
    sk_id_cust = int(request.args.get('SK_ID_CURR'))
    # get data of 20 nearest neigh in the X_tr_featsel dataframe (pd.DataFrame)
    neigh = NearestNeighbors(n_neighbors=20)
    neigh.fit(X_train)
    X_cust = X_train.loc[sk_id_cust: sk_id_cust]
    idx = neigh.kneighbors(X=X_cust,
                           n_neighbors=20,
                           return_distance=False).ravel()
    nearest_cust_idx = list(X_train.iloc[idx].index)
    X_neigh_df = X_train.loc[nearest_cust_idx, :]
    y_neigh = y_train.loc[nearest_cust_idx]
    return X_neigh_df, y_neigh

# return data of 20 neighbors of one customer when requested (SK_ID_CURR)
# Test local : http://127.0.0.1:5000/api/neigh_cust/?SK_ID_CURR=425217

@app.route('/api/neigh_cust/')
def neigh_cust():
    # Parse the http request to get arguments (sk_id_cust)
    sk_id_cust = int(request.args.get('SK_ID_CURR'))
    # return the nearest neighbors
    X_neigh_df, y_neigh = get_df_neigh(sk_id_cust)
    # Convert the customer's data to JSON
    X_neigh_json = json.loads(X_neigh_df.to_json())
    y_neigh_json = json.loads(y_neigh.to_json())
    # Return the cleaned data jsonified
    return jsonify({'status': 'ok',
    				'X_neigh': X_neigh_json,
    				'y_neigh': y_neigh_json})

# return all data of training set when requested
# Test local : http://127.0.0.1:5000/api/all_proc_data_tr/
# Test Heroku : https://oc-api-flask-mm.herokuapp.com/api/all_proc_data_tr/
@app.route('/api/all_proc_data_tr/')
def all_proc_data_tr():
    # get all data from X_tr_featsel, X_te_featsel and y_train data
    # and convert the data to JSON
    X_tr_featsel_json = json.loads(X_train.to_json())
    y_train_json = json.loads(y_train.to_json())
    # Return the cleaned data jsonified
    return jsonify({'status': 'ok',
    				'X_tr_proc': X_tr_featsel_json,
    				'y_train': y_train_json})


@app.route('/api/features_desc/', methods=['GET'])
# Test : http://127.0.0.1:5000/api/features_desc?SK_ID_CURR=399070
def send_features_descriptions():
    SK_ID_CURR = int(request.args.get('SK_ID_CURR'))
    
    # Caractéristiques pour les statistiques descriptives
    desc_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'ANNUITY_INCOME_PERCENT', 'DAYS_EMPLOYED_PERCENT', 'CREDIT_TERM']

    # Sélection du client par son index (SK_ID_CURR)
    client_index = SK_ID_CURR  # Replace `client_id` with the value of the client you want to find the index of
    # Remplacez SK_ID_CURR par l'index du client souhaité

    # Calcul des statistiques descriptives pour les clients similaires
    X_test_id
    X_test_id.fillna(X_test_id.mean(), inplace=True)
    sim_desc = X_test_id.loc[X_test_id.index != client_index, desc_features].describe().reset_index()
    sim_desc.rename(columns={'index': 'Statistiques'}, inplace=True)
    sim_desc['Groupe'] = 'Clients similaires'

    # Calcul des statistiques descriptives pour le client sélectionné
    client_desc = X_test_id.loc[client_index, desc_features].describe().reset_index()
    client_desc.rename(columns={'index': 'Statistiques'}, inplace=True)
    client_desc['Groupe'] = 'Client sélectionné'

    # Concaténation des deux DataFrames
    features_desc= pd.concat([sim_desc, client_desc], axis=0)
    features_desc = features_desc.reset_index()

    # Converting the pd.Series to JSON
    features_desc_json = json.loads(features_desc.to_json())
    
    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'data': features_desc_json
     })


@app.route('/api/features_imp/', methods=['GET'])
# Test : http://127.0.0.1:5000/api/features_imp
def send_features_importance():
    features_importance = pd.Series(lgbm_model.feature_importances_, index=lgbm_model.feature_name_)

    # Converting the pd.Series to JSON
    features_importance_json = json.loads(features_importance.to_json())

    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'data': features_importance_json
     })


@app.route('/api/local_interpretation/', methods=['GET'])
# Test : http://127.0.0.1:5000/api/local_interpretation?SK_ID_CURR=399070


def send_local_interpretation():
    # Parsing the HTTP request to get arguments (applicant ID)
    SK_ID_CURR = int(request.args.get('SK_ID_CURR'))
    ind=dic_indices[SK_ID_CURR]
    # Getting the personal data for the applicant (pd.DataFrame)
    local_data = X_test_id.loc[SK_ID_CURR:SK_ID_CURR]

    # Compute the SHAP values for the local data
    
    
    # Extract the SHAP values for the first instance and create a pd.Series
    shap_values_flat = np.array(shap_values[0][ind]).flatten()
    features_contribs = pd.Series(shap_values_flat[:12], index=local_data.columns[:12])
    print(features_contribs)

    # Convert the prediction to a regular int
    prediction = int(lgbm_model.predict(local_data)[0])

    # Convert the pd.Series to a dictionary and convert int64 to int
    features_contribs_dict = features_contribs.to_dict()
    features_contribs_dict = {k: v for k, v in features_contribs_dict.items()}

    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'prediction': prediction,
        'contribs': features_contribs_dict,
    })







#################################################
if __name__ == "__main__":
    app.run(debug=True)
