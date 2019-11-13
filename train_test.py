# Load libraries.
from ibm_watson import VisualRecognitionV3 # pip install ibm_watson
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
          print("Training model", model_name)
          model = visual_recognition.create_classifier(
              name = model_name,
              positive_examples = {'stress': stress},
              negative_examples = ns).get_result()
          return(model)
        except Exception as ex:
          print("Failed to train model", model_name)
          print("Error: ", ex)

# Wait until model is trained.
def wait(modelID):
    # Loop indefinitely.
    while True:
        # Get status.
        classifier = visual_recognition.get_classifier(modelID).get_result()
        status = classifier['status']
        # If ready, break loop.
        if status == 'ready':
            print(modelID, 'ready')
            break
        # If training, wait 30 seconds and try again.
        if status == 'training':
            print('Pinged', modelID, time.ctime())
            time.sleep(30)
            continue
        # If failed, print explanation.
        if status == 'failed':
            print(modelID, 'failed')
            print(classifier['explanation'])
            break
        # Temp.
        print(modelID)
        print(status)

# Test zip.
def test_zip(modelID, test_path):
    with open(test_path, 'rb') as images_file:
        response = visual_recognition.classify(
            images_file,
            threshold = '0',
            classifier_ids = [modelID]).get_result()
    return(response)

# Convert response to dataframe.
def res_to_df(response):
    # Create empty dataframe for storing output.
    df = []
    # Loop through images.
    for img in response['images']:
        # Get image information.
        file = img['image']
        model = img['classifiers'][0]['classifier_id']
        score = img['classifiers'][0]['classes'][0]['score']
        # Append to dataframe.
        df.append({'image': file, 'model': model, 'score': score})
    # Convert list to dataframe and return.
    df = pd.DataFrame(df)
    return(df)

# Wrapper function tests positive and negative test sets and returns
# concatenated dataframe.
def test_multiple(modelID, stress_test, ns_test):
    print("Testing model", modelID)
    # Test stress images.
    stress_res = test_zip(modelID, stress_test)
    stress_df = res_to_df(stress_res)
    stress_df['stress'] = True
    # Test no_stress images.
    ns_res = test_zip(modelID, ns_test)
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
        df = pd.read_csv(path, index_col=0)
        df = df.append(dic, ignore_index = True)
    # If csv doesn't exist, convert dictionary to dataframe.
    else:
        df = pd.DataFrame.from_dict(dic, orient='index').T
    # Save dataframe.
    df.to_csv(path)

# Wrapper function trains model and returns modelID.
def pipeline_train(model_name, stress_train, ns_train, log = 'Results/Log.csv'):
    #
    # Args:
    #   model_name: what to name the model.
    #   stress_train, ns_train: paths to zip files containing training set images.
    #   log: path to csv file to which model information will be appended.
    #
    # Train model.
    modelID = train_model(model_name, ns_train, stress_train)['classifier_id']
    # Update training log.
    log = {'model': modelID, 'stress_train': stress_train, 'ns_train': ns_train,
           'time': time.ctime()}
    save(log, 'Results/Log.csv')
    # Return modelID.
    return(modelID)
    
# Wrapper function tests model and returns performance metrics.
def pipeline_test(modelID, stress_test, ns_test, pred_save = 'Results/'):
    #
    # Args:
    #   modelID: modelID returned by 'pipeline_train'.
    #   stress_test, ns_test: paths to zip files containing test set images.
    #   pred_save: path to directory where image-level predictions will be saved.
    #   trained: path to directory where images will be moved to after use.
    #
    # Test model.
    pred = test_multiple(modelID, stress_test, ns_test)
    # Save predictions.
    full_path = os.path.join(pred_save, modelID + '.csv')
    pred.to_csv(full_path)
    # Return predictions.
    return(pred)

def pipeline_assess(pred, stress_test, ns_test, perf_save = 'Results/Performance.csv'):
    #
    # Args:
    #   pred: image-level predictions return by pipeline_test.
    #   stress_test, ns_test: paths to zip files containing test set images.
    #   perf_save: path to csv file to which model performance metrics will be appended.
    #
    # Calculate performance metrics.
    perf = performance(pred)
    # Append image paths and timestamp.
    perf['stress_test'] = stress_test
    perf['ns_test'] = ns_test 
    perf['time'] = time.ctime()
    # Append to 'perf_save' csv.
    save(perf, perf_save)
    # Return performance metrics.
    return(perf)

