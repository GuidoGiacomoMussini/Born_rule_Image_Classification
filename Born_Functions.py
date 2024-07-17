## Born Classes

import numpy as np
from tqdm import tqdm as progress_bar
import cv2
import time
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import f1_score #, confusion_matrix, 
from sklearn.model_selection import GridSearchCV, train_test_split
from scipy.spatial.distance import cdist
import pickle

class Feature_Extraction:

  def __init__(self):
    pass

  def convert_to_grayscale(self, image):
    '''
    convert the images in grayscale.
      input: RGB Image d(32x32x3)
      Output: Grayscale Image d(32x32)
    '''
    try: 
      return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    except: 
      return image

  def extract_descriptors_from_image(self, image, extractor_dict):
    '''
    extract the features of an image using a given extractor:
      Input: grayscale image d(32, 32)
      Output: descriptor np.array d(extractor_len)
    '''
    extractor = extractor_dict['extractor']
    # exctract the features using the extractors
    kpoints, descriptors = extractor.detectAndCompute(image, None)

    if descriptors is None:
        return None, None
    
    return descriptors, kpoints
  
  def extract_descriptors_from_data(self, images, labels, extractor_dict): 
    X, y, to_remove= [], [], []

    for index, image in progress_bar(enumerate(images), desc= "extract descriptors:"): 
      #extract descriptors from an image
      descriptors, _ = self.extract_descriptors_from_image(image, extractor_dict)

      #store descriptors if != None
      if descriptors is None:
        to_remove.append((image, labels[index]))
      else:
        X.append(descriptors)
        y.append(labels[index])
    
    print(f"removed {len(to_remove)} empty descriptors")

    return X, y, to_remove


  def build_visual_vocabulary(self, descriptors_list, k):
    '''
    Build the visual vocabulary by clustering the descritptors found in all the train set.
      Input: descriptor list d(extractor_len, len(training_set))
      Output: visual vocab d(k, descriptor_len)
    '''
    # vertically stack the descriptor list
    features_reshaped = np.vstack(descriptors_list)

    # apply cluster algorithm to derive the visual words
    minibatch_kmeans = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=100, n_init=3)
    minibatch_kmeans.fit(features_reshaped)

    return minibatch_kmeans

  # def build_histograms(self, descriptors_list, visual_vocab, distance_metric):
  #   '''
  #   Create the histograms of each image, given the distance metric.
  #     Input: descriptors_list d(extractor_len, len(dataset))
  #            visual vocab d(k, descriptor_len)
  #     Output: histograms d(len(dataset), k)
  #   '''

  #   # retrieve k from the vocabulary
  #   k = visual_vocab.shape[0]
  #   # initialize the histograms
  #   histograms = np.zeros((len(descriptors_list), k))

  #   # skip the images without descriptors
  #   for i, descriptors in enumerate(descriptors_list):
  #     if np.all(descriptors == 0):
  #       continue
  #     # distance between descriptors and visual words
  #     distances = cdist(descriptors, visual_vocab, metric=distance_metric)
  #     # assign each descriptor to the nearest visual word, given the distance metric
  #     closest_clusters = np.argmin(distances, axis=1)
  #     # create the 'histogram' for each image
  #     for cluster_idx in closest_clusters:
  #         histograms[i][cluster_idx] += 1

  #   return histograms

  # def build_histograms(self, descriptors_list, visual_vocab, distance_metric):
  #     k = visual_vocab.shape[0]
  #     histograms = np.zeros((len(descriptors_list), k))

  #     for i, descriptors in enumerate(descriptors_list):
  #         if not np.any(descriptors):
  #             continue
  #         distances = cdist(descriptors, visual_vocab, metric=distance_metric)
  #         closest_clusters = np.argmin(distances, axis=1)
  #         histograms[i] = np.bincount(closest_clusters, minlength=k)

  #     return histograms

  def build_histograms(self, df, visual_vocab, distance_metric):
      k = visual_vocab.shape[0]
      histograms = np.zeros((len(df), k))

      # vw list
      visual_words_list = []

      for i, row in progress_bar(df.iterrows()):
          #extract descriptors from each image
          descriptors = row['descriptors']
          #handle empty descriptors
          if descriptors.size == 0:
              visual_words_list.append([])
              continue
          
          #assign each descriptor to closest cluster
          distances = cdist(descriptors, visual_vocab, metric=distance_metric)
          closest_clusters = np.argmin(distances, axis=1)
          
          # store assigned vw in list -> for df
          visual_words_list.append(closest_clusters.tolist())

          # populate histogram
          histograms[i] = np.bincount(closest_clusters, minlength=k)
      
      # assign assigned_vw to df
      df['assigned_vw'] = visual_words_list
      
      return histograms, df

class Test_Pipeline:

  def __init__(self, classifiers, distance_metrics, extractors, k_values):
    
    self.FE = Feature_Extraction()
    self.tuning_result = None
    self.classifiers = classifiers
    self.supported_distance_metrics = distance_metrics
    self.supported_extractors = extractors
    self.k_values = k_values


  def check_dataset(self, X, y):
    '''
    checks that the dataset is in the correct form
    '''
    assert isinstance(X, np.ndarray) and X.ndim == 4, "X has to be an np.array with 4 dimensions: (Num images, Height, Width, Channels)"
    assert isinstance(y, np.ndarray) and y.ndim == 2, "y has to be an np.array with 2 dimensions: (Num labels, 1)"
    assert X.shape[0] == y.shape[0], "The number of examples in X is different from the number of examples in y: %d vs %d" % (X.shape[0], y.shape[0])

  def run_configuration(self, X, y, test_size, extractor_, k, distance_metric_, classifier_):
    '''
    test a given configuration of: classifier, extractor, distance measure, vocab size.
    '''
    self.check_dataset(X, y)

    # grayscale
    grayscale_images = np.array([self.FE.convert_to_grayscale(img) for img in X])

    # extract the descriptors
    extractor_dict = self.supported_extractors[extractor_]
    descriptors_list = [self.FE.extract_descriptors(img, extractor_dict) for img in grayscale_images]

    # create train and test set
    descriptors_train, descriptors_test, y_train, y_test = train_test_split(descriptors_list, y, test_size=test_size, random_state=19)

    # build visual vocabulary
    visual_vocab = self.FE.build_visual_vocabulary(descriptors_train, k)

    # create the histograms
    distance_metric = self.supported_distance_metrics[distance_metric_]
    histograms_train = self.FE.build_histograms(descriptors_train, visual_vocab, distance_metric)
    histograms_test = self.FE.build_histograms(descriptors_test, visual_vocab, distance_metric)

    # train the classifier
    classifier = self.classifiers[classifier_]
    classifier.fit(histograms_train, y_train.ravel())

    # test the classifier
    predictions = classifier.predict(histograms_test)

    # store the results
    results = {'y_obs': y_test, 'y_pred': predictions}

    return results

  def run_test(self, X, y, filename, test_size=0.2):
    '''
    run different configurations of classifier, extractor, distance measure, vocab size.
    '''
    tuning_results = []

    for classifier_name in progress_bar(self.classifiers.keys()):
      for extractor_name in progress_bar(self.supported_extractors.keys()):
        for distance_metric in progress_bar(self.supported_distance_metrics.keys()):
          for k in progress_bar(self.k_values):
            start_time = time.time()
            results = self.run_configuration(X, y, test_size, extractor_name, k, distance_metric, classifier_name)
            end_time = time.time()
            f1 = f1_score(results['y_obs'], results['y_pred'], average='macro')
            comp_time = end_time - start_time

            tuning_results.append({
                'Params': {'classifier': classifier_name, 'extractor': extractor_name, 'distance_metric': distance_metric, 'k': k},
                'F1 Score': f1*100,
                'Computation Time': comp_time
                #'Prediction_Results': results
                })
    try: 
      with open(filename, "wb") as file:
              pickle.dump(tuning_results, file)
    except: 
      raise ValueError(f'the file {filename} has not been saved')
              
    return tuning_results