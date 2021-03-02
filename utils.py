
from matplotlib import pyplot
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
##Let's compare the number of words in the fake post and real post
## and try to distinguish pattern in the fake and real post based on number of words used in the post.

 ## 'company_profile' -  fake post has less words in the company profile while real post has more words.

## 'description' - Both the post has similar distribution of words in description.
## And see that the words are more focused as in company profile.

## 'requirements' - The distribution of words in requirements of the fake and real post are similar.
## And see that the words are more focused as in company profile.

## 'benefits' - The distribution of words in benefits of the fake and real post are similar
def plot_frequency_words_comp(df):

    col = ['company_profile', 'description', 'requirements', 'benefits']
    n_col = ['company_profile_len', 'description_len', 'requirements_len', 'benefits_len']
    i = 0
    j = 0
    while (i < len(col)):
        df[n_col[j]] = df[col[i]].str.split().map(lambda x: len(repr(x)))
        pyplot.figure(figsize=(15, 5), dpi=100)

        df[df['fraudulent'] == 0][n_col[j]].plot(bins=35, kind='hist', color='blue',
                                                         label='Real Job Post', alpha=0.6)
        df[df['fraudulent'] == 1][n_col[j]].plot(kind='hist', color='red',
                                                         label='Fake Job Post', alpha=0.6)
        pyplot.legend()
        pyplot.xlabel("Text Length")
        pyplot.title(f'Words in {col[i]}')
        pyplot.show()

        fig, (ax1, ax2) = pyplot.subplots(ncols=2, figsize=(17, 5), dpi=100)
        num = df[df["fraudulent"] == 1][col[i]].str.split().map(lambda x: len(repr(x)))
        ax1.hist(num, bins=20, color='red')
        ax1.set_title('Fake Post')
        num = df[df["fraudulent"] == 0][col[i]].str.split().map(lambda x: len(repr(x)))
        ax2.hist(num, bins=20)
        ax2.set_title('Real Post')
        fig.suptitle(f'Words in {col[i]}')
        pyplot.show()

        i += 1
        j += 1

## Visualize job postings by countries
def visualize_by_countries(df):
    country_tmp = df.groupby(['country_code', 'fraudulent'])['location'].count().reset_index()
    country_tmp = country_tmp.sort_values(by='location', ascending=False)
    pyplot.figure(figsize=(15, 5))
    sns.barplot(x='country_code', y='location', hue='fraudulent', data=country_tmp[:10])
    pyplot.title('10th countries most posted job')
    pyplot.show()

def visualize_wordclouds(df):

    ## Separate fraud and actual jobs
    fraudjobs_text = df[df.fraudulent == 1].text
    actualjobs_text = df[df.fraudulent == 0].text

    ## Fraudulent jobs word cloud
    pyplot.figure(figsize=(16, 14))
    wc = WordCloud(min_font_size=3, max_words=3000, width=1600, height=800, stopwords=STOPWORDS).generate(
        str(" ".join(fraudjobs_text)))
    pyplot.title('Fraud Cloud')
    pyplot.imshow(wc, interpolation='bilinear')
    pyplot.show()
    # plt.savefig('fraud_cloud.jpeg')

    ## Actual jobs wordcloud
    pyplot.figure(figsize=(16, 14))
    wc = WordCloud(min_font_size=3, max_words=3000, width=1600, height=800,
                   stopwords=STOPWORDS).generate(str(" ".join(actualjobs_text)))
    pyplot.title('Genuine Cloud')
    pyplot.imshow(wc, interpolation='bilinear')
    pyplot.show()
    # pyplot.savefig('no_fraud_cloud.jpeg')


def fraud_real_data_visualization(df):
    ## Fraud and Real visualization
    sns.countplot(df['fraudulent']).set_title('Real & Fradulent')
    df.groupby('fraudulent').count()['text'].reset_index().sort_values(by='text', ascending=False)
    pyplot.show()


def accuracy_comparison(comparison):
    keys = comparison.keys()
    values = comparison.values()

    pyplot.figure(figsize=(10, 5))
    pyplot.plot(keys, values, marker='o')
    pyplot.ylim(80, 100)
    pyplot.xlabel('classifier name')
    pyplot.ylabel('accuracy')
    pyplot.title('Model Comparison')
    pyplot.grid()
    pyplot.show()
