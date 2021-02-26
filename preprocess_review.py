import csv
import os
import re
from pyltp import SentenceSplitter
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import Parser
from datetime import datetime


LTP_DATA_DIR = 'pyltp-resource'
cws_model_path = os.path.join(LTP_DATA_DIR, 'ltp-model/cws.model')
pos_model_path = os.path.join(LTP_DATA_DIR, 'ltp-model/pos.model')
par_model_path = os.path.join(LTP_DATA_DIR, 'ltp-model/parser.model')
cut_text_path = os.path.join(LTP_DATA_DIR, 'word_segmentation.txt')
stop_text_path = os.path.join(LTP_DATA_DIR, 'stop_words.txt')

segmentor = Segmentor()
segmentor.load_with_lexicon(cws_model_path, cut_text_path)


def get_stopwords_list(filepath=stop_text_path):
  stopwords = [x.strip() for x in open(filepath, 'r').readlines()]
  return stopwords


def review_process(review, stopwords):
  # split the whole sentence
  review = re.sub(' +', '', review)
  sentences = SentenceSplitter.split(review)
  # print(review)

  new_text = []
  for sent in sentences:
    # word segmentation
    # print(sent)
    sent1 = re.sub('[^0-9A-Za-z\u4e00-\u9fa5]', ' ', sent.strip())
    words = list(segmentor.segment(sent1))
    # print(sent1, words)
    # remove stop words
    remain_text = [item for item in words if item not in stopwords]
    if len(remain_text) > 0:
      new_text.append(sent)
  
  new_text = ''.join(new_text)
  if new_text.startswith('，'):
    new_text = new_text[1:]
    
  return new_text


def process(input_file, output_file):
  stop_words = get_stopwords_list()
  all_reviews = []
  all_date = []
  with open(input_file, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for line in reader:
      date = line[1].strip()
      rate = line[3].strip().split('.')[0]
      
      # process each text
      title = re.sub('[^0-9A-Za-z\u4e00-\u9fa5]', '', line[4])
      content = line[5].strip()
      content = content.replace('\\', '')
      content = content.replace(',', '，')
      content = content.replace('.', '。')
      content = re.sub('-{3,}', '', content)
      text = title + '，' + content
      if '该条评论已经被删除' in text or len(text) > 5000:
        continue
      if "开发者回复" in text:
        text = text[:text.index('开发者回复')]
      
      text_processed = review_process(text, stop_words)
      if len(text_processed) != 0:
        review_line = '-*-'.join([text_processed, date, rate])
        if review_line not in all_reviews:
          all_reviews.append(review_line)
          all_date.append(datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))
        else:
          print('duplicated review text: ' + review_line)
  
  # sort by date and write to files
  all_reviews_sorted = [y for _, y in sorted(zip(all_date, all_reviews))]
  with open(output_file, 'w', encoding='utf-8') as file:
    for line in all_reviews_sorted:
      file.write(line)
      file.write('\n')

