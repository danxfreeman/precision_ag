'''
Author : Shaurya Gupta
Permissions : Use : Anyone
              Edit :  with consent of Author
              
Date : 2/21/2019

Following dependencies need to be installed :
    GraphQL : pip install graphqlclient
    OpenCV : pip install opencv-python
    
Note : Remember to import query.
    
'''
import numpy as np
import glob
import json
import time
import os
import urllib
import cv2
#from query import query 
from graphqlclient import GraphQLClient
client = GraphQLClient('https://api.labelbox.com/graphql')
client.inject_token('eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjanMwd3I4em1idWc0MDg5OGx5Z3RlMWczIiwib3JnYW5pemF0aW9uSWQiOiJjanMwd3I4ejFib2NnMGI0NzBmejY2cDJzIiwiYXBpS2V5SWQiOiJjanM0cWludW9nYW56MGE2Mm03YnpqYzRrIiwiaWF0IjoxNTUwMTU1NDI4LCJleHAiOjIxODEzMDc0Mjh9.u5HkyUF7jcgGsc6uG_ogBocUrV_5NeKJzoD1P7lnDaA')

class query:
    
    def __init__(self):
        None
        
    def labels(self, id):
        
        target = 'mutation{exportLabels(data:{projectId: ' + "\"" + str(id) + "\"" + '}){ createdAt downloadUrl shouldPoll }}'
        return target
    
    project_list = """query MyProjects {   user{   projects{   id  name}}}"""

def get_id():
    
    q = query()
    
    res_str = client.execute(q.project_list)

    res = json.loads(res_str)
    
    return res

def get_url(id):
    
    q = query()
    
    res_str = client.execute(q.labels(id))

    res = json.loads(res_str)
    
    url_data = res['data']['exportLabels']
    
    return url_data

def get_labels(id):
    
    url_data = get_url(id)
    
    if url_data['shouldPoll'] :
        print("URL link is being generated...")
        time.sleep(3)
        return get_labels(id)
    
    with urllib.request.urlopen(url_data['downloadUrl']) as url:
        labels = json.loads(url.read().decode())
    return labels

def work_on_data(data):
    
    #plants = list(data[0]['Label'].keys())
    final = []
    j = 0
    for i in range(len(data)):
        if data[i]['Label'] == 'Skip':
                continue
        plants = list(data[i]['Label'].keys())
        for plant in plants:
            for label in data[i]['Label'][plant]:
                j = j + 1
                x,y = points(label)
                final.append([data[i]['Dataset Name']\
                         ,data[i]['External ID']\
                         ,plant\
                         ,label['condition']\
                         ,min(x)\
                         ,min(y)\
                         ,max(y) - min(y)\
                         ,max(x) - min(x)\
                         ,j])
                         
    return final

def crop_image(data):
    #x, y, h, w
    
    # dictionary mislabelled
#    condition = {'low_water_stress' : 'hws', 
#          'high_water_stress' : 'lws',
#          'no_stress' : 'ns',
#          'unknown' : 'ps'
#         }
    condition = {'low_water_stress' : 'lws', 
          'high_water_stress' : 'hws',
          'no_stress' : 'ns',
          'unknown' : 'ps'
         }
    
    # abbreviate species (used dict 'condition' instead of 'folder')
#    folder = {'low_water_stress' : 'Low Water Stress', 
#              'high_water_stress' : 'High Water Stress',
#              'no_stress' : 'Not stressed',
#              'unknown' : 'Unknown'
#             }
    
    species = {'Cornus Obliqua': 'co',
               'Hydrangeo paniculata': 'hp',
               'Hydrangeo quercifolia': 'hq',
               'Buddleia': 'bud',
               'Physocarpus opulifolius': 'po',
               'Spiraea japonica': 'sj'}
    
    x = 4; y = 5; h = 6; w = 7
    #i = 1

    path = r'/Users/danielfreeman/Desktop/ag/Images/'

    for label in data[:]:
        if not os.path.isdir(path + label[2]):
            os.mkdir(path + label[2])
        #image = cv2.imread(path + '/Images/' +str(label[1]),1)
        image = cv2.imread(path + str(label[1]), 1)
        crop_img = image[label[y]:label[y] + label[h], label[x]:label[x] + label[w]]
        #filename = label[0] + '_' + label[1].split('.')[0] + '_' + condition[label[3]] + '_' + str(i) + '.JPG'
        filename = label[1].split('.')[0] + '_' + species[label[2]] + '_' + condition[label[3]] + '_' + str(label[8]) + '.JPG' # identify using last column in 'final' instead of i
        if not os.path.isdir(path + label[2] + '/' + condition[label[3]]):
            os.mkdir(path + label[2] + '/' + condition[label[3]])
        cv2.imwrite(path + label[2] + '/' + condition[label[3]] + '/' + filename,crop_img)
        #print(crop_img.shape, ' ', label[1], ' ', label[2])
        #i = i + 1

def segment_images():
    
    r = 50
    c = 50
    
    condition = {'LWS' : 'Low Water Stress', 
                  'HWS' : 'High Water Stress',
                  'NS' : 'Not stressed',
                  'PS' : 'Unknown'
                 }
        
    os.makedirs('Segmented_Images/Cornus obliqua/High Water Stress')
    os.makedirs('Segmented_Images/Cornus obliqua/Low Water Stress')
    os.makedirs('Segmented_Images/Cornus obliqua/Unknown')
    os.makedirs('Segmented_Images/Cornus obliqua/Not stressed')
    os.makedirs('Segmented_Images/Hydrangeo quercifolia/High Water Stress')
    os.makedirs('Segmented_Images/Hydrangeo quercifolia/Low Water Stress')
    os.makedirs('Segmented_Images/Hydrangeo quercifolia/Unknown')
    os.makedirs('Segmented_Images/Hydrangeo quercifolia/Not stressed')
    os.makedirs('Segmented_Images/Hydrangeo paniculata/High Water Stress')
    os.makedirs('Segmented_Images/Hydrangeo paniculata/Low Water Stress')
    os.makedirs('Segmented_Images/Hydrangeo paniculata/Unknown')
    os.makedirs('Segmented_Images/Hydrangeo paniculata/Not stressed')
    os.makedirs('Segmented_Images/Spiraea japonica/High Water Stress')
    os.makedirs('Segmented_Images/Spiraea japonica/Low Water Stress')
    os.makedirs('Segmented_Images/Spiraea japonica/Unknown')
    os.makedirs('Segmented_Images/Spiraea japonica/Not stressed')
    os.makedirs('Segmented_Images/Buddleia/High Water Stress')
    os.makedirs('Segmented_Images/Buddleia/Low Water Stress')
    os.makedirs('Segmented_Images/Buddleia/Unknown')
    os.makedirs('Segmented_Images/Buddleia/Not stressed')
    os.makedirs('Segmented_Images/Physocarpus opulifolius/High Water Stress')
    os.makedirs('Segmented_Images/Physocarpus opulifolius/Low Water Stress')
    os.makedirs('Segmented_Images/Physocarpus opulifolius/Unknown')
    os.makedirs('Segmented_Images/Physocarpus opulifolius/Not stressed')
    
    total = 1
    for file in glob.iglob('Cropped_Images/*/*/*.JPG'):
       
        image = np.array(cv2.imread(file,1))
        
        try:
            resize_img = cv2.resize(image,(150,150))
        except:
            continue
        finally:
            crop_num = 1
            for i in range(0, 150 - r, r):
                for j in range(0, 150 - c, c):
                    crop = resize_img[j:j + c,i:i + r]
                    tokens = file.split('\\')[3].split('_')
                    cv2.imwrite('./Segmented_Images/' + file.split('\\')[1] + '/' + condition[tokens[4]] + '/' + \
                                tokens[0] + '_' + tokens[3] + '_' + tokens[4] + '_' + tokens[5].split('.')[0] + \
                                str(crop_num) + '.JPG', crop)                                               
                    print(str(total) + ' - ' + str(crop_num))
                    crop_num = crop_num + 1
                    
            total = total + 1
            
            

def points(label):
    x = []
    y = []
    for i in range(0,4):
        x.append(label['geometry'][i]['x'])
        y.append(label['geometry'][i]['y'])
    return(x,y)    
    
       

if __name__ == "__main__":

  user_label = get_labels("cjs0wso1cnxr40b29vrr8ooty")
  final = work_on_data(user_label)
  crop_image(final)
  #segment_images()
  
  
import csv

with open("/Users/danielfreeman/Desktop/ag/coord.csv", 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for i in range(1, len(final)):
        wr.writerow(final[i])
  
  
  
  
  
  
  
  