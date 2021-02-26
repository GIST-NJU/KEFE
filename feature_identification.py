"""
The key feature identification tool
"""
import argparse
import subprocess
import csv
import os
from datetime import datetime
from key_feature import KeyFeature

bert_path = 'bert-master'
temp_dir = 'temp'


def produce_matching_file(feature_path, review_path):
  features = []
  with open(feature_path, 'r', encoding='utf-8') as file:
    for r in file.readlines():
      features.append(r.strip())
      
  reviews = []
  with open(review_path, 'r', encoding='utf-8') as file:
    for r in file.readlines():
      reviews.append(r.strip())
  
  # prepare the test data
  test_file = temp_dir + '/match_test_set.tsv'
  with open(test_file, 'w', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter='\t')
    for r in reviews:
      for f in features:
        writer.writerow([0, f, r.split('-*-')[0]])
  
  # do prediction
  print('[INFO] running prediction of user review matching ...')
  print('       # features = {}, # reviews = {}'.format(len(features), len(reviews)))
  subprocess.run('python3 {}/run_classifier.py'
                 ' --task_name=match'
                 ' --do_predict=true'
                 ' --data_dir={}'
                 ' --vocab_file={}/chinese_L-12_H-768_A-12/vocab.txt'
                 ' --bert_config_file={}/chinese_L-12_H-768_A-12/bert_config.json'
                 ' --init_checkpoint={}/model-2'
                 ' --max_seq_length=128'
                 ' --output_dir={}'.format(bert_path, test_file, bert_path, bert_path, bert_path, temp_dir),
                 shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  os.remove(temp_dir + '/predict.tf_record')
  print('[INFO] prediction result of classifier: {}/test_results.tsv'.format(temp_dir))
  
  predicted = []
  with open('{}/test_results.tsv'.format(temp_dir), 'r', encoding='utf-8') as file:
    for row in csv.reader(file, delimiter='\t'):
      l0, l1 = float(row[0]), float(row[1])
      predicted.append('0' if l0 > l1 else '1')
  
  # generate the matching file
  result_file = temp_dir + '/match_feature_review.txt'
  with open(result_file, 'w', encoding='utf-8') as file:
    index = 0
    for r in reviews:
      for f in features:
        file.write(f + '-*-' + r + '-*-' + predicted[index] + '\n')
        index += 1
  print('[INFO] the matching results between features and user reviews are generated: {}'.format(result_file))

  return features, result_file
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Identify key features of the app based on the given features and user '
                                               'reviews')
  parser.add_argument('-f', metavar='FEATURE_FILE', type=str, required=True,
                      help='the file of specified app features (.txt)')
  
  # the MATCH_FILE should be ordered by date
  parser.add_argument('-m', metavar='MATCH_FILE', type=str,
                      help='the matching results between features and reviews (.txt); if this file is provided, '
                           'the REVIEW_FILE will be ignored')
  parser.add_argument('-r', metavar='REVIEW_FILE', type=str,
                      help='the file of user reviews (.txt)')
  
  parser.add_argument('-o', metavar='OUTPUT_FILE', type=str, default='key_features.txt',
                      help='the key features identified (default: key_features.txt)')
  parser.add_argument('--bert', type=str, default=bert_path,
                      help='path of bert directory (default: {})'.format(bert_path))

  args = parser.parse_args()
  feature_file = args.f
  match_file = args.m
  review_file = args.r
  output_file = args.o
  bert_path = args.bert = args.bert

  if match_file is None:
    # if there is no matching file, generate this file first
    features, match_file = produce_matching_file(feature_file, review_file)
  else:
    features = []
    with open(feature_file, 'r', encoding='utf-8') as file:
      for line in file.readlines():
        features.append(line.strip())
    print('[INFO] use the existing user review matching file: {}'.format(match_file))
  
  # get the latest date and date interval
  feature_reviews = []
  all_dates = []
  with open(match_file, 'r', encoding='utf-8') as file:
    for line in file.readlines():
      feature_reviews.append(line)
      line = line.strip().split('-*-')
      all_dates.append(datetime.strptime(line[2].split(' ')[0], '%Y-%m-%d'))
      
  all_dates = sorted(all_dates)
  earliest_date = all_dates[0]
  latest_date = all_dates[-1]
  interval = (latest_date - earliest_date).days + 1
  print('[INFO] perform analysis with reviews posted between {} and {} ({} days)'.format(
    earliest_date.strftime('%Y-%m-%d'), latest_date.strftime('%Y-%m-%d'), interval))

  # perform key feature analysis
  k = KeyFeature(features, feature_reviews)
  _, _, key_features = k.key_feature_identification(latest_date)
  feature_str = [e[1] for e in key_features]
  print('[INFO] identify {} key features (see {}):'.format(len(key_features), output_file))
  print('       {}'.format(feature_str))
  
  with open(output_file, 'w', encoding='utf-8') as file:
    for e in feature_str:
      file.write(e)
      file.write('\n')
  
  print('[TASK FINISHED]')
