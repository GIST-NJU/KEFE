# KEFE

KEFE is an approach that exploits the information of app description and user reviews (written in Chinese) to identify **key features** that have a significant relationship with app rating scores. 

The application of KEFE involves three main steps: 1) applying a textual pattern-based approach and a deep machine learning classifier to extract features from app description; 2) applying another classifier to match features with their relevant user reviews; and 3) applying regression analysis to identify key features. More details can be found in the following paper:

> Huayao Wu, Wenjun Deng, Xintao Niu, and Changhai Nie. Identifying Key Features from App User Reviews. International Conference on Software Engineering (ICSE), 2021



### Usage

KEFE is developed and tested using `Python 3`, `pyltp` and `tensorflow`. Please install the following packages of specific versions:

```
pip install pyltp=0.2.1
pip install tensorflow=1.15.0
```

More instructions for installing  `pyltp` and `tensorflow` can be found in their respective websites: [pyltp](https://github.com/HIT-SCIR/pyltp), [tensorflow](https://github.com/tensorflow/tensorflow).



1. Download the model files, which include:

   * pyltp model files: [ltp-model](https://github.com/GIST-NJU/KEFE/releases/download/v1.0/ltp-model.zip)
   * pre-trained BERT model: [chinese_L-12_H-768_A-12](https://github.com/GIST-NJU/KEFE/releases/download/v1.0/chinese_L-12_H-768_A-12.zip)
   * classification model of feature extraction: [model-extract](https://github.com/GIST-NJU/KEFE/releases/download/v1.0/model-extract.zip)
   * classification model of user review matching: [model-match](https://github.com/GIST-NJU/KEFE/releases/download/v1.0/model-match.zip)

   The `ltp-model` should be put into the `pyltp-resource` directory, and the other three should be put into the `bert-master` directory.



2. To extract feature-describing phrases from a given app description, run:

   ```bash
   python feature_extraction.py -i [app_description].csv
   # for example
   # python feature_extraction.py -i example/description.csv
   ```

   

3. To identify key features of a given app, run:

   ```bash
   python feature_identification.py -f [features].txt -r [reviews].txt
   # for example
   # python feature_identification.py -f example/alipay_features.txt -r example/alipay_reviews.txt
   ```

   The above command will first apply the classification model to match features and user reviews (this will take a long time if there is a large volume of user reviews), and then identify key features. If an existing file of matching between features and user reviews is available, run:

   ```bash
   python feature_identification.py -f [features].txt -m [matching].txt
   # for example
   # python feature_identification.py -f example/alipay_features.txt -m example/alipay_matching.txt
   ```

   **Format of Files**

   * The `[reviews].txt` file should be organised as the following format per line: `[review_text]-*-[review_date]-*-[rating_score]`
   * The `[matching].txt` file should be organised as the following format per line: `[feature]-*-[review_text]-*-[review_date]-*-[rating_score]-*-[label]`, where label = `0` and `1` indicate non-matching and matching pairs, respectively.



### Experimental Results

To be updated.


### Dataset

To be updated.
