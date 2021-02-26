import os
import csv
from pyltp import SentenceSplitter
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import Parser
import re
import sys

content_index = 5

end_str_list = [
  '与我们交流', '联系我们', '欢迎联系', '本次更新', '官方网站', '客服电话', '客服热线', '全国热线', '官方微信', 'http',
  'www', '意见反馈。', '联系方式。', '更新内容:', '微博:', 'QQ群：', '官方客服', '邮箱：', '微信公众号：', '帮助与反馈',
  '官方微博：', '官网：', '全国热线', '热线：', '微信公众平台：', '近期更新：', '联系我们：', '连续包月', '订阅服务'
]

VOB_SBV_COO_str_list = [
  '是', '为', '免', '无需', '在', '达', '无', '等', '愿', '让', '求', '例如', '带', '多', '荣获', '落', '跨', '包',
  '用', '帮', '开通', '上线', '运用', '融入', '没有', '涵盖', '包括', '覆盖', '知''方便', '集成', '确保', '预设',
  '包含', '云集', '超过', '超', '靠', '给', '成', '变', '欢迎', '贯穿', '起', '没', '掉', '破', '出品', '想', '讲',
  '说', '集', '小', '有', '拥有', '到', '关爱', '结合', '来', '爱', '还有', '及时达', '当', '变化', '背书', '告别',
  '无惧', '力求', '按', '吃', '满足', '助', '搜罗', '解放', '趴', '希望', '包括', '保', '互通', '请', '推出', '发育',
  '孕育', '还是', '怀', '成为', '见证', '翻开', '翻', '遍布', '形成', '扩大', '致力', '拒绝', '沉淀', '耕耘', '通过',
  '享受', '懂', '变', '做到'
]

individual_feature_words = [
  '登录', '注册', '摇一摇', '扫一扫', '朋友圈', '直播', '搜索', '电话', '短信', '打卡', '审批', '汇报', '单聊', '群聊',
  '收款', '红包', '预约', '预订', '付款', '打车', '支付', '定位', '导航', '分享', '打车', '下单', '外卖', '改签', '退票',
  '打车', '补票', '抢票', '打车', '聊天', '购物', '转账', '信用住', '听说读写', '音标', '发音', '词性', '释义', '签到',
  '用法', '搭配', '换肤', '答题', '已读未读'
]


class CandidatePhraseExtractor:
  def __init__(self, input_file, output_file, pyltp_path=None):
    # ltp model files
    LTP_DATA_DIR = 'pyltp-resource/ltp-model' if pyltp_path is None else pyltp_path
    cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
    par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')
    cut_text_path = os.path.join(LTP_DATA_DIR, 'word_segmentation.txt')

    self.input_file = input_file
    self.output_file = output_file
    self.raw_data = []
    self.sentence_cutted = []             # 分词后list，每一个子元素为一个app描述分词后的list
    self.postagger = Postagger()          # 初始化词性标注实例
    self.postagger.load(pos_model_path)   # 加载词性标注模型
    self.parser = Parser()                # 初始化句法分析实例
    self.parser.load(par_model_path)      # 加载句法分析模型
    self.segmentor = Segmentor()
    self.segmentor.load_with_lexicon(cws_model_path, cut_text_path)

  @staticmethod
  def remove_serial_number(text):
    text = re.sub('\d[,.、]', '。', text)
    text = re.sub('[●,=√#￥%&，【】◆ ★◎☆⊙]', '。', text)
    return text

  @staticmethod
  def remove_exce_punc(text):
    duels = [x + x for x in list('。，,=！!-—#')]
    for d in duels:
      while d in text:
        text = text.replace(d, d[0])
    return text

  def read_data_from_file(self):
    with open(self.input_file, encoding='utf-8') as f:
      reader = csv.reader(f)
      for row in reader:
        self.raw_data.append(row)

  def get_seg_sentence_from_pyltp(self):
    for sentence in self.raw_data:
      # 去除作为编号的字符
      sent_temp = self.remove_serial_number(sentence[content_index])
      sent_temp_2 = self.remove_exce_punc(sent_temp)
      sents = SentenceSplitter.split(sent_temp_2)
      temp_list = []  # 保存一个app描述的处理结果
      for sent in sents:  # 一个app描述中分句后的所有句子
        # 判断app描述中是否有句子包含end_str, 若有，直接跳过该app描述中该句子之后的所有内容
        skip_sent = False
        for end_str in end_str_list:
          if end_str in sent:
            skip_sent = True
            break
        if skip_sent is True and len(sent) < 400:  # len(sent) < 400：由于分句错误而将整个app评视为一句
          break
        # 对每个句子进行分词
        words = self.segmentor.segment(sent)
        temp_list.append(list(words))  # list(words) 为一句话的分词结果 ['应用', '介绍', '：', ',', '1.', '可以', '发'...]
      self.sentence_cutted.append(temp_list)  # 每个子list为一个app描述的处理结果
    self.segmentor.release()

  def get_postage_and_parser(self):
    function_phrase = []  # 记录所有候选功能短语的list
    for sentences in self.sentence_cutted:
      app_phrase = []  # 保存一个app中候选功能短语的list
      for sentence in sentences:
        if sentence[-1] in ['。', '！', '?']:
          sentence = sentence[0:-1]
        postags = self.postagger.postag(sentence)  # 词性标注
        postage_list = list(postags)
        arcs = self.parser.parse(sentence, postage_list)  # 句法分析
        S = set()
        sent_phrase = []
        VOB_exist = False
        for i in range(len(arcs)):
          # if arcs[i].relation == 'ATT':
          #   ATT_exist = True
          if arcs[i].relation == 'VOB':
            VOB_exist = True
        # ATT_object_index_max = 0
        # if ATT_exist:
        #   # ATT_object_index_max = 0
        #   for i in range(len(arcs)):
        #     if arcs[i].relation == 'ATT':
        #       if ATT_object_index_max < arcs[i].head - 1:
        #         ATT_object_index_max = arcs[i].head - 1

        for i in range(len(arcs)):
          # pattern 10
          if sentence[i] in individual_feature_words:  # 单个能表示功能的词
            app_phrase.append(sentence[i])
          if postage_list[i] == 'wp':
            continue
          temp = arcs[i].head - 1
          str1 = sentence[i]
          str2 = sentence[temp]
          temp2 = arcs[temp].head - 1
          str3 = sentence[temp2]

          if not VOB_exist:  # and not COO_exist
            if arcs[i].relation == 'ATT':
              if sentence[temp] not in S:  # 控制每个句子中被ATT修饰的, 相同sentence[temp]只输出一次
                S.add(sentence[temp])
              else:
                continue
              if sentence[i] not in S:  # 控制每个句子中每个ATT只输出一次
                S.add(sentence[i])
              else:
                continue
              # 这里输出所有与sentence[temp]具有定中关系的修饰词及sentence[temp]
              # pattern 2
              for j in range(len(arcs)):
                if j < i:
                  continue
                elif arcs[j].relation == 'ATT' and arcs[j].head - 1 == temp:
                  # 修饰当前 ATT 的 ATT
                  if arcs[j - 2].relation == 'ATT' and arcs[j - 2].head - 1 == j:
                    sent_phrase.append(sentence[j - 2])
                  if arcs[j - 1].relation == 'ATT' and arcs[j - 1].head - 1 == j:
                    sent_phrase.append(sentence[j - 1])
                  sent_phrase.append(sentence[j])  # 当前ATT
              sent_phrase.append(str2)
              if arcs[temp].relation == 'ATT' or arcs[
                temp].relation == 'ADV':  # 安全 管理 企业 数据 ，安全管理为指向 企业ATT，而企业指向 数据
                sent_phrase.append(str3)
              app_phrase.append(sent_phrase)
              sent_phrase = []

          # pattern 1 and pattern 3
          # different patterns can be extracted in the same way
          if arcs[i].relation == 'VOB':  # 输出语法依存分析中 动宾 关系
            if str2 in VOB_SBV_COO_str_list:
              continue
            sent_phrase.append(str2)
            for j in range(len(arcs)):
              if arcs[j].relation == 'ATT' and arcs[j].head - 1 == i:  # 该词用以修饰宾语，如 发文字消息 中的文字
                if arcs[j - 2].relation == 'ATT' and arcs[j - 2].head - 1 == j:
                  sent_phrase.append(sentence[j - 2])
                if arcs[j - 1].relation == 'ATT' and arcs[j - 1].head - 1 == j:
                  sent_phrase.append(sentence[j - 1])
                sent_phrase.append(sentence[j])
            sent_phrase.append(str1)
            app_phrase.append(sent_phrase)
            sent_phrase = []

          # supplement for pattern 1 and pattern 3
          # different patterns can be extracted in the same way
          elif arcs[i].relation == 'SBV':  # 输出语法依存分析中 主谓 关系， 由于分词、词性标注、句法分析的错误而必须考虑
            if str2 in VOB_SBV_COO_str_list:
              continue
            for j in range(len(arcs)):
              if arcs[j].relation == 'ATT' and arcs[j].head - 1 == i:  # 该词用以修饰 谓语
                if arcs[j - 2].relation == 'ATT' and arcs[j - 2].head - 1 == j:
                  sent_phrase.append(sentence[j - 2])
                if arcs[j - 1].relation == 'ATT' and arcs[j - 1].head - 1 == j:
                  sent_phrase.append(sentence[j - 1])
                sent_phrase.append(sentence[j])
            for k in range(i, temp + 1):  # 输出i到temp之间的所有词
              sent_phrase.append(sentence[k])
            sent_phrase = []

          # pattern 5 and pattern 6 and pattern 8 and pattern 9
          # different patterns can be extracted in the same way
          elif arcs[i].relation == 'COO':  # 输出语法依存分析中 并列 关系
            if arcs[temp].relation == 'VOB':
              if sentence[temp2] in VOB_SBV_COO_str_list:
                continue
              sent_phrase.append(sentence[temp2])
              for j in range(len(arcs)):
                if arcs[j].relation == 'ATT' and arcs[j].head - 1 == i:
                  sent_phrase.append(sentence[j])
              sent_phrase.append(sentence[i])
              app_phrase.append(sent_phrase)
              sent_phrase = []

          # pattern 4
          elif arcs[i].relation == 'FOB':  # 输出语法依存分析中 前置宾语 关系
            sent_phrase.append(str2)
            for j in range(len(arcs)):
              if arcs[j].relation == 'ATT' and arcs[j].head - 1 == i:
                sent_phrase.append(sentence[j])
              if arcs[j].relation == 'ATT' and arcs[j].head - 1 == temp:
                sent_phrase.append(sentence[j])
            sent_phrase.append(str1)
            app_phrase.append(sent_phrase)
            sent_phrase = []

          else: # DBL、RAD、HED ...
            for j in range(len(arcs)):
              if arcs[j].relation == 'FOB' and arcs[j].head - 1 == i:
                sent_phrase.append(sentence[temp])
                for k in range(len(arcs)):
                  if arcs[k].relation == 'ATT' and arcs[k].head - 1 == j:  # 该词用以修饰宾语，如 发文字消息 中的文字
                    sent_phrase.append(sentence[k])
                sent_phrase.append(sentence[j])
                app_phrase.append(sent_phrase)
                sent_phrase = []

              # pattern 7
              elif arcs[j].relation == 'VOB' and arcs[j].head - 1 == i:
                if arcs[i - 1].relation == ('WP' or 'ADV'):
                  continue
                sent_phrase.append(sentence[i])
                for k in range(len(arcs)):
                  if arcs[k].relation == 'ATT' and arcs[k].head - 1 == j:  # 该词用以修饰宾语
                    sent_phrase.append(sentence[k])
                sent_phrase.append(sentence[j])
                app_phrase.append(sent_phrase)
                sent_phrase = []
          if arcs[i - 1].relation == 'ATT' and arcs[i - 1].head == i + 1:
            for j in range(len(arcs)):
              if arcs[j].relation == 'ATT' and arcs[j].head - 1 == i - 1:  # 该词用以修饰当前词
                sent_phrase.append(sentence[j])
            sent_phrase.append(sentence[i - 1])
            sent_phrase.append(str1)
            app_phrase.append(sent_phrase)
            sent_phrase = []
        S.clear()
      function_phrase.append(app_phrase)
    self.postagger.release()
    self.parser.release()

    # 去除重复项
    final_function_phrase = []
    for app_phrase in function_phrase:
      for sent_phrase in app_phrase:
        sent_phrase = ''.join(sent_phrase)
        sent_phrase = re.sub('[^0-9A-Za-z\u4e00-\u9fa5]', '', sent_phrase)
        if sent_phrase != '' and sent_phrase not in final_function_phrase:
          final_function_phrase.append(sent_phrase)

    # 结果写入tsv文件中
    train_set_file = open(self.output_file, 'w', newline='')
    csv.register_dialect('tsv_dialect', delimiter='\t')
    writer = csv.writer(train_set_file, dialect='tsv_dialect')
    for final_app_phrase in final_function_phrase:
      writer.writerow([1, final_app_phrase])
    train_set_file.close()


if __name__ == '__main__':
  # extract the set of candidate phrases from the given app description file
  input_file = sys.argv[1]
  output_file = sys.argv[2]

  test = CandidatePhraseExtractor(input_file, output_file)
  test.read_data_from_file()
  test.get_seg_sentence_from_pyltp()
  test.get_postage_and_parser()
