import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en import English
import utils
import string

## remove warnings
import warnings
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

class DataProccess:

    def __init__(self):
        self.df = pd.read_csv('./data/fake_job_postings.csv')
        self.data = None

    def clean_data(self):
        print("Analyze data before cleaning..")

        utils.plot_frequency_words_comp(self.df)

        print("Initializing data clean up..")

        self.df.interpolate(inplace=True)

        data2 = self.df.copy()

        ## I do not need these columns for classification
        data2.drop(['salary_range', 'job_id', 'department', 'telecommuting','has_company_logo', 'has_questions'], axis=1, inplace=True)

        data2 = data2.sort_values('title').reset_index(drop=True)

        ## fill the NaN values that are present in the pandas dataframe.
        data2['employment_type'] = data2['employment_type'].bfill(axis=0)
        data2['required_experience'] = data2['required_experience'].bfill(axis=0)
        data2['required_education'] = data2['required_education'].bfill(axis=0)
        data2['industry'] = data2['industry'].bfill(axis=0)
        data2['function'] = data2['function'].bfill(axis=0)
        data2['benefits'] = data2['benefits'].bfill(axis=0)
        data2['location'] = data2['location'].bfill(axis=0)

        data2['country_code'] = data2['location'].str.split(',', expand=True)[0]

        data3 = data2.copy()
        ## Return a boolean same-sized object indicating if the values are not NaN (1 = NaN, 0 = not NaN)
        data3 = data3[data3['description'].notna()]

        ## Drop the columns where any of the elements is NaN.
        data3 = data3.dropna(axis=0, how='any')

        ## Drop duplicates except for the first occurrence.
        data3 = data3.drop_duplicates(keep='first')

        data4 = data3.copy()

        ## Visualize job postings by countries
        utils.visualize_by_countries(data4)


        ## Drop the columns where the elements are NaN.
        data4.dropna(inplace=True)
        data4.drop(['location'], axis=1, inplace=True)
        data4['text'] = data4['description'] + ' ' + data4['title'] + ' ' + \
                        data4['employment_type'] + ' ' + data4['required_experience'] + ' ' + \
                        data4['required_education'] + ' ' + data4['industry'] + ' ' + \
                        data4['function'] + ' ' + data4['requirements'] + ' ' + data4['company_profile'] + ' ' + \
                        data4['country_code'] + ' ' + data4['benefits']

        del data4['description']
        del data4['title']
        del data4['employment_type']
        del data4['required_experience']
        del data4['required_education']
        del data4['industry']
        del data4['function']
        del data4['requirements']
        del data4['company_profile']
        del data4['country_code']
        del data4['benefits']

        utils.visualize_wordclouds(data4)

        ## Fraud and Real visualization
        utils.fraud_real_data_visualization(data4)

        self.data = data4.copy()

        return self.data


    def prepare_data(self):

        data_clean = self.clean_data()

        ## Create our list of stopwords
        nlp = spacy.load('en_core_web_sm')
        stop_words = spacy.lang.en.stop_words.STOP_WORDS

        ## Load English tokenizer, tagger, parser, NER and word vectors
        parser = English()

        STOPLIST = set(stop_words)
        ## Unite a punctuation mark and unite with another space
        SYMBOLS = " ".join(string.punctuation).split(" ")

        def tokenizetext(sample):
            ## Replacing a new line with a space
            text = sample.strip().replace("\n", " ")

            ## Uppercase to lowercase
            text = text.lower()

            #3 Creating our token object, which is used to create
            ## documents with linguistic annotations.
            tokens = parser(text)
            lemmas = []

            ## Lemmatizing each token and converting each token into lowercase
            for tok in tokens:
                lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
            tokens = lemmas

            ## Removing stop words
            tokens = [tok for tok in tokens if tok not in STOPLIST]
            tokens = [tok for tok in tokens if tok not in SYMBOLS]

            return tokens

        ## min_df = When building the vocabulary ignore terms that have a document frequency
        ## strictly lower than the given threshold
        print("Tokenizing text..")
        vectorizer = TfidfVectorizer(tokenizer=tokenizetext, ngram_range=(1, 3), min_df=0.06)

        print("Transforming data..")
        vectorizer_features = vectorizer.fit_transform(data_clean['text'])


        print("Producing features and labels..")
        labels = data_clean['fraudulent']
        features = vectorizer_features.todense()

        print(pd.DataFrame(features, columns=vectorizer.get_feature_names()))
        return features, labels

