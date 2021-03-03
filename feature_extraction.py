"""
The feature extraction tool
"""
import argparse
import subprocess
import csv
import os
from candidate_phrase import CandidatePhraseExtractor
from pathlib import Path

bert_path = 'bert-master'
pyltp_resource_path = 'pyltp-resource/ltp-model'
phrase_file = 'temp/candidate_phrase.tsv'
temp_dir = 'temp'
Path(temp_dir).mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Extract features from the given app description.')
  parser.add_argument('-i', metavar='INPUT_FILE', type=str, required=True,
                      help='the app description file (.csv)')
  parser.add_argument('-o', metavar='OUTPUT_FILE', type=str, default='features.txt',
                      help='the file of features extracted (default: features.txt)')
  parser.add_argument('--pyltp', type=str, default=pyltp_resource_path,
                      help='path of pyltp resource directory (default: {})'.format(pyltp_resource_path))
  parser.add_argument('--bert', type=str, default=bert_path,
                      help='path of bert directory (default: {})'.format(bert_path))
  
  args = parser.parse_args()
  input_file = args.i
  output_file = args.o
  pyltp_resource_path = args.pyltp
  bert_path = args.bert = args.bert
  
  # extract candidate phrases
  test = CandidatePhraseExtractor(input_file, phrase_file, pyltp_path=args.pyltp)
  test.read_data_from_file()
  test.get_seg_sentence_from_pyltp()
  test.get_postage_and_parser()
  print('[INFO] candidate phrases extracted: {}'.format(phrase_file))
  
  # determine whether each phrase is feature describing
  subprocess.run('python3 {}/run_classifier.py'
                 ' --task_name=extract'
                 ' --do_predict=true'
                 ' --data_dir={}'
                 ' --vocab_file={}/chinese_L-12_H-768_A-12/vocab.txt'
                 ' --bert_config_file={}/chinese_L-12_H-768_A-12/bert_config.json'
                 ' --init_checkpoint={}/model-extract'
                 ' --max_seq_length=128'
                 ' --output_dir={}'.format(bert_path, phrase_file, bert_path, bert_path, bert_path, temp_dir),
                 shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  os.remove(temp_dir + '/predict.tf_record')
  print('[INFO] prediction result of classifier: {}/test_results.tsv'.format(temp_dir))
  
  # get the final feature describing phrases
  candidate_phrase = []
  with open(phrase_file, encoding='utf-8') as file:
    for row in csv.reader(file, delimiter='\t'):
      candidate_phrase.append(row[1])
  
  writer = open(output_file, 'w', encoding='utf-8')
  with open(temp_dir + '/test_results.tsv', encoding='utf-8') as file:
    index = 0
    for row in csv.reader(file, delimiter='\t'):
      p1, p2 = float(row[0]), float(row[1])   # labels 1 and 2
      if p1 > p2:
        writer.write(candidate_phrase[index])
        writer.write('\n')
      index += 1
  writer.close()
  print('[INFO] the set of features: {}'.format(output_file))
  print('[TASK FINISHED]')
