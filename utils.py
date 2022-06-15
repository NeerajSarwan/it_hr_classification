import re
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from langdetect import detect
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
stop_words.update(['ee','p','g','de','hi','call','number','e','–','cc','n','c','hello','I','________________________________','r'])

lemmatizer = nltk.stem.WordNetLemmatizer()

def cleanPunc(sentence):
    cleaned = re.sub(r'-',r'',sentence)
    cleaned = re.sub(r'\S*\d+\S*',r' ',cleaned)
    cleaned = re.sub(r'\[.*?\]',r' ',cleaned)
    cleaned = re.sub(r'[?|!|\'|"|#]',r' ',cleaned)
    cleaned = re.sub(r'[A-Za-z0-9]*@[A-Za-z]*\.?[A-Za-z0-9]*',r" ",cleaned)
    cleaned = re.sub(r'[.|,|)|(|\|/|:|=|;|•|*|>|&|-|+]',r' ',cleaned)
    cleaned = cleaned.replace("\n | \r"," ")
    return cleaned

def cleanHtml(sentence):
    cleaned = re.compile('<.*?>')
    cleaned = re.sub(cleaned,' ',str(sentence))
    return cleaned

def remove_whitespaces(sentence):
    return " ".join(sentence.split())

def removeStopWords(sentence):
    re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
    return re_stop_words.sub(" ",sentence)

def cleaning_func(sentence):
    cleaned = removeStopWords(sentence)
    cleaned = cleanHtml(cleaned)
    cleaned = cleanPunc(cleaned)
    cleaned = remove_whitespaces(cleaned)
    cleaned = stemming(cleaned)
    return cleaned

def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = lemmatizer.lemmatize(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence


class IT_HR_Classifier():
    def __init__(self):
        self.model = LogisticRegression()
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def _preprocess(self, df, mode="inference"):
        """
        df : pandas dataframe
        
        This functions combines short_description and long_description column to complete_description
        and then cleans the complete_description column to generate clean text for feature extraction.
        It also drops rows where there is empty complete_description
        
        Expected columns : short_description, long_description
        
        Returns : Numpy array
        """
        if len(set(["short_description", "long_description"]) - set(df.columns)) > 0:
            raise ValueError("Either short_description or long_description column not present in df. Please check column names")
        else:
            df["complete_description"] = df["short_description"] + " " + df["long_description"]
            df.drop(["short_description", "long_description"], axis=1, inplace=True)
            df['lang'] = df['complete_description'].apply(lambda x: detect(x))
            df = df.loc[df['lang']=="en"].drop(["lang"], axis=1)
            if df.shape[0] == 0:
                raise ValueError("No English text detected.")
            else:
                df["complete_description"] = df["complete_description"].apply(lambda x: cleaning_func(x))
                df["num_chars"] = df["complete_description"].apply(lambda x: len(x))
                df = df.loc[df['num_chars']>=2].drop(["num_chars"], axis=1)
                if df.shape[0] == 0:
                    raise ValueError("Only 1 character detected. Insufficient for Inference")
            return df
        
    def _extract_features(self, df):
        """
        df: pandas dataframe
        
        This function creates numeric features using Sentence transformers for the downstream classification task.
        
        Expected columns : complete_description
        
        Returns : Numpy array
        """
        if "complete_description" not in df.columns:
            raise ValueError("complete_description column not present.")
        else:
            sentences = list(df['complete_description'].values)
            embeddings = self.sentence_model.encode(sentences)
            return embeddings

    def fit(self, x, y):
        print("Train : Pre-processing short_description and long_description")
        x = self._preprocess(x)
        y = y[x.index]
        print("Train: Computing Sentence Transformer embeddings")
        x = self._extract_features(x)
        y = np.array(y)
        print("Training Model")
        self.model.fit(x, y)
    
    def predict(self, x):
        print("Test : Pre-processing short_description and long_description")
        x = self._preprocess(x, "inference")
        print("Test: Computing Sentence Transformer embeddings")
        x = self._extract_features(x)
        print("Generating Predictions")
        predictions = self.model.predict(x)
        return predictions