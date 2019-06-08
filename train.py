# Load libraries.
#pip install ibm_watson
from ibm_watson import VisualRecognitionV3
import os
import pandas as pd
import numpy as np
import time
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# Access cloud.
visual_recognition = VisualRecognitionV3(
    version = '2018-03-19',
    iam_apikey = 'uAuf4MQpv4sHFcqjfdlnMLx2RGqpM1OIyGQsGklZ8M7L'
)

# Train model.
def train_model(model_name, ns_train, stress_train):
    with open(ns_train, 'rb') as ns, open(stress_train ,'rb') as stress:
        try:
          print("Training model:", model_name)
          model = visual_recognition.create_classifier(
              name = model_name,
              positive_examples = {'stress': stress, 'no_stress': ns}).get_result()
          return(model)
        except Exception as ex:
          print("Failed to train model:", model_name)
          print("Error: ", ex)

# Wait until model is trained.
def wait(modelID):
    # Loop indefinitely or until status changes.
    status = 0
    while status == 0:
        # Get status of all models.
        classifiers = visual_recognition.list_classifiers().get_result()
        # Get status of specific model.
        for c in classifiers['classifiers']:
            if c['classifier_id'] == modelID:
                # If ready, break loop.
                if c['status'] == 'ready':
                    status = 1
                    print('Model ready')
                # If not ready, wait 30 seconds and try again.
                else:
                    print('Pinged', modelID, time.ctime())
                    time.sleep(30)

# Test zip.
def test_images(modelID, test_path):
    print('Testing', modelID)
    with open(test_path, 'rb') as images_file:
        response = visual_recognition.classify(
            images_file,
            threshold = 0 ,
            classifier_ids = [modelID]).get_result()
    return(response)

# Convert response to dataframe.
def res_to_df(response):
    # Create empty dataframe for storing output.
    df = []
    # Loop through images.
    for img in response['images']:
        # Get identifying information.
        file = img['image']
        model = img['classifiers'][0]['name']
        # Get score for stress class.
        result = img['classifiers'][0]['classes']
        for c in result:
            if c['class'] == 'stress':
                score = c['score']
        # Append to dataframe..
        df.append({'image': file, 'model': model, 'score': score})
    # Convert list to dataframe and return.
    df = pd.DataFrame(df)
    return(df)

# Wrapper function tests positive and negative test sets and returns
# concatenated dataframe.
def test_multiple(modelID, stress_test, ns_test):
    print("Testing model:", modelID)
    # Test stress images.
    stress_res = test_images(modelID, stress_test)
    stress_df = res_to_df(stress_res)
    stress_df['stress'] = True
    # Test no_stress images.
    ns_res = test_images(modelID, ns_test)
    ns_df = res_to_df(ns_res)
    ns_df['stress'] = False
    # Concatenate dataframes.
    df = pd.concat([stress_df, ns_df])
    return(df)

# Calculate performance metrics.
def performance(df):
    # Get identifying information.
    modelID = df.model.iloc[0]
    # Get true class and score for each instance.
    y_true = df.stress
    y_score = df.score
    # Calculate f_score for each threshold.
    f_scores = []
    thresholds = np.linspace(0.001,0.999,50)
    for t in thresholds:
        y_pred = y_score > t
        f = f1_score(y_true, y_pred)
        f_scores.append(f)
    # Get optimal f_score and threshold.
    f = max(f_scores)
    i = f_scores.index(f)
    t = thresholds[i]
    # Get precision, recall, and ROC-AUC.
    p = precision_score(y_true, y_score > t)
    r = recall_score(y_true, y_score > t)
    roc = roc_auc_score(y_true, y_score)
    # Count instances.
    n_stress = sum(y_true)
    n_ns = len(df) - n_stress
    # Return dictionary.
    perf = {'model': modelID, 'N_stress': n_stress, 'N_no_stress': n_ns,
            'opt_threshold': t, 'precision': p, 'recall': r, 'f_score': f,
            'roc_auc': roc}
    return(perf)

# Append dictionary to csv file.
def save(dic, path):
    # If csv exists, open as dataframe and append.
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df.append(dic, ignore_index = True)
    # If csv doesn't exist, convert dictionary to dataframe.
    else:
        df = pd.DataFrame.from_dict(dic, orient = 'index').T
    # Save dataframe.
    df.to_csv(path)

# Wrapper function train, tests, and assesses model.
def pipeline(model_name, stress_train, ns_train, stress_test, ns_test,
             pred_save, perf_save, trained, parent = os.getcwd()):
    #
    # Args:
    #   stress_train, ns_train, stress_test, ns_test: paths to zip files
    #   containing test and training sets for stress and no stress conditions.
    #   predictions: path to directory where image-level predictions will be saved.
    #   performance: path to csv file to which model performance will be appended.
    #   trained: path to directory where images will be moved to after use.
    #   parent: working directory.
    #
    # Change directory.
    os.chdir(parent)
    # Train model.
    modelID = train_model(model_name, ns_train, stress_train)['classifier_id']
    # Wait until model is trained.
    wait(modelID)
    # Test model and save.
    pred = test_multiple(modelID, stress_test, ns_test)
    # Save dataframe.
    full_path = os.path.join(pred_save, modelID + '.csv')
    pred.to_csv(full_path)
    # Assesss performance.
    perf = performance(pred)
    return(perf)
    # Append image paths and timestamp.
    perf['stress_train'] = stress_train
    perf['ns_train'] = ns_train
    perf['stress_test'] = stress_test
    perf['ns_test'] = ns_test 
    perf['time'] = time.ctime()
    # Save as csv.
    save(perf, perf_save)

# Bud: NS v. HWS.
pipeline(model_name = 'MAPIR_bud_HWS_k1',
         parent = '/Users/danielfreeman/Desktop/ag/',
         stress_train = 'split/MAPIR_FLT2_Buddleia_high_water_stress_k1_train.zip',
         ns_train = 'split/MAPIR_FLT2_Buddleia_no_stress_k1_train.zip',
         stress_test = 'split/MAPIR_FLT2_Buddleia_high_water_stress_k1_test.zip',
         ns_test = 'split/MAPIR_FLT2_Buddleia_no_stress_k1_test.zip',
         pred_save = 'Results/',
         perf_save = 'Results/performance.csv')

## Bud: NS v. S.
#pipeline(model_name = 'MAPIR_bud_WS_k1',
#         parent = '/Users/danielfreeman/Desktop/ag/',
#         stress_train = 'split2/MAPIR_FL2_Buddleia_stress_train_k1',
#         ns_train = 'split2/MAPIR_FL2_Buddleia_no_stress_train_k1',
#         stress_test = 'split2/MAPIR_FL2_Buddleia_stress_test_k1',
#         ns_test = 'split2/MAPIR_FL2_Buddleia_no_stress_test_k1',
#         pred_to = 'Results/',
#         perf_to = 'Results/performance.csv')
#
## All: NS v. S
#pipeline(model_name = 'MAPIR_all_WS_k1',
#         parent = '/Users/danielfreeman/Desktop/ag/',
#         stress_train = 'split3/MAPIR_FL2_pool_stress_train_k1',
#         ns_train = 'split3/MAPIR_FL2_pool_no_stress_train_k1',
#         stress_test = 'split3/MAPIR_FL2_pool_stress_test_k1',
#         ns_test = 'split3/MAPIR_FL2_pool_no_stress_test_k1',
#         pred_to = 'Results/',
#         perf_to = 'Results/performance.csv')
#
#pipeline(model_name = '',
#         parent = '',
#         stress_train = '',
#         ns_train = '',
#         stress_test = '',
#         ns_test = '',
#         pred_to = '',
#         perf_to = '')
#        




