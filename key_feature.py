import numpy as np
import datetime
import statsmodels.api as sm


def get_features(file_name, app):
  with open(file_name, 'r', encoding='utf-8') as file:
    for line in file.readlines():
      line = line.strip().split(',')
      if line[0] == app:
        return line[1:]


def get_annotated_reviews(file_name):
  with open(file_name, 'r', encoding='utf-8') as file:
    return file.readlines()


class KeyFeature:
  def __init__(self, features, reviews, day=180, time_unit=1):
    self.feature_index = 0
    self.review_content_index = 1
    self.date_index = 2
    self.rating_index = 3
    self.label_index = 4
    self.split_symbol = '-*-'
    self.date_str = '%Y-%m-%d %H:%M'
  
    self.features = features    # a list of features
    self.reviews = reviews      # a list of annotated feature_review pairs
    self.day = day
    self.time_unit = time_unit
    
    # process annotated review file
    self.all_feature_reviews = self.get_feature_review_list(self.features, self.reviews)
    
  def get_feature_review_list(self, features, reviews):
    """
    Determine the set of features that is mentioned in each review. The result will be used as the
    input of key feature identification.
    :return: a list of data instances [0, review_text, review_date, review_rating, feature_set]
    """
    # assign an unique ID for each feature (start from 0)
    feature_dict = {}
    for i, feature in enumerate(features):
      feature_dict[feature] = i
  
    # determine the set of features that is mentioned in each review
    final_list = []
    for i in range(0, len(reviews), len(features)):
      review = reviews[i].strip().split(self.split_symbol)
      feature_label = []
      for j in range(len(features)):
        review = reviews[i + j].strip().split(self.split_symbol)
        if review[self.label_index] == '1':  # this is a match
          f = str(feature_dict[review[self.feature_index]])
          feature_label.append(f)
      
      feature_label_str = ','.join(feature_label) if len(feature_label) > 0 else '-1'
      final_list.append([0, review[self.review_content_index], review[self.date_index],
                         review[self.rating_index], feature_label_str])
  
    return final_list

  def key_feature_identification(self, current_date):
    """
    Identify key features based on both positive and negative reviews, and return the combination of their results.
    The reviews posted in the previous [self.day] days will be used to perform analysis.
    """
    # filter reviews based on date
    bad_reviews = []
    good_reviews = []
    start_date = current_date - datetime.timedelta(days=self.day)
    for review in self.all_feature_reviews:
      date = review[self.date_index]
      date = datetime.datetime.strptime(date, self.date_str)
      if start_date <= date < current_date:
        if review[self.rating_index] in ['1', '2']:  # negative reviews
          bad_reviews.append(review)
        elif review[self.rating_index] in ['4', '5']:  # positive reviews
          good_reviews.append(review)
        
    positive_key_features, num1, num_sum1 = self.get_key_feature(current_date, good_reviews)
    negative_key_features, num2, num_sum2 = self.get_key_feature(current_date, bad_reviews)
    
    # combine key features of positive and negative reviews
    positive_key_features.sort(reverse=True)
    negative_key_features.sort(reverse=True)
    
    key_feature_dict = {}
    for x in positive_key_features + negative_key_features:
      if x[1] in key_feature_dict:
        key_feature_dict[x[1]] += float(x[0])
      else:
        key_feature_dict[x[1]] = float(x[0])

    all_key_features = []
    for key in key_feature_dict:
      all_key_features.append([key_feature_dict[key], key])
    all_key_features.sort(reverse=True)
    
    return positive_key_features, negative_key_features, all_key_features
  
  def get_key_feature(self, current_date, feature_reviews_used):
    """
    Identify the set of key features based on the specific feature_review data.
    """
    start_date = current_date - datetime.timedelta(days=self.day)
    time_unit_numbers = int(self.day / self.time_unit)
    
    # calculate the number of reviews in each time unit
    num = np.zeros((len(self.features), time_unit_numbers), dtype=np.int32)
    num_sum = np.zeros(time_unit_numbers, dtype=np.int32)
    for review in feature_reviews_used:
      date = datetime.datetime.strptime(review[self.date_index], self.date_str)
      index = int((date - start_date).days / self.time_unit)
      assert 0 <= index < self.day
      num_sum[index] += 1
      # for each feature mentioned
      for s in review[self.label_index].split(','):
        assert s != ''
        if s != '-1':
          num[int(s), index] += 1
    
    # remove features that are not mentioned in any review
    feature_index = []
    for k in range(len(self.features)):
      if np.sum(num[k]) != 0:
        feature_index.append(k)
    num1 = num[feature_index, :]
    feature_str = np.array(self.features)[feature_index]
    
    # regression analysis
    x = np.column_stack(num1)   # rows = day, columns = number of features
    x2 = sm.add_constant(x, prepend=True)
    y = num_sum
    est = sm.OLS(y, x2).fit()
    
    sig_features_index = []
    for i, p in enumerate(est.pvalues):
      if i > 0 and p < 0.05:
        sig_features_index.append(i - 1)
    sig_feature_str = feature_str[sig_features_index]
    sig_feature_coef = []
    
    # adjust the coefficients
    if len(sig_feature_str) > 0:
      num2 = num[sig_features_index, :]
      x = np.column_stack(num2)
      x2 = sm.add_constant(x, prepend=True)
      est_new = sm.OLS(y, x2).fit()
      
      for i in range(1, len(est_new.params)):
        sig_feature_coef.append(est_new.params[i])
    
    return [[x, y] for x, y in zip(sig_feature_coef, sig_feature_str)], num, num_sum

