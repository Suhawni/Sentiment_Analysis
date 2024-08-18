# %%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objs as go
import plotly.graph_objs as go
import plotly.offline as py

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# %%
mcd = pd.read_csv('C:\\Users\\hp\\Downloads\\Reviews2.csv\\Reviews.csv', encoding="latin-1")

# %%
import nltk
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# %%
# Performing sentiment analysis on each review
sentiments = []
for review in mcd['Text']:
    sentiment = sia.polarity_scores(review)
    sentiments.append(sentiment)

# %%
sentiment_labels = []
label=[]
for sentiment in sentiments:
    compound_score = sentiment['compound']
    if compound_score >= 0.05:
        sentiment_labels.append('Positive')
        label.append('1')
    elif compound_score <= -0.05:
        sentiment_labels.append('Negative')
        label.append('-1')
    else:
        sentiment_labels.append('Neutral')
        label.append('0')

# %%
mcd['sentiment'] = sentiment_labels
mcd['label'] = label

# %%
mcd[['Text', 'sentiment', 'label']]

# %%
X = mcd['Text']
y = mcd[['sentiment','label']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming you have already imported the necessary libraries and have your data loaded
# Your code for creating X and y goes here

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DataFrames for training and testing data
train_df = pd.DataFrame({'Text': X_train, 'sentiment': y_train['sentiment'], 'label': y_train['label']})
test_df = pd.DataFrame({'Text': X_test})


# %%
print(f'Train data shape: {train_df.shape}')
train_df.head()

# %%
print(f'Test data shape: {test_df.shape}')
test_df.head()

# %%
train_df.duplicated().sum()
test_df.duplicated().sum()

# %%
test_df = test_df.drop_duplicates()
train_df = train_df.drop_duplicates()

# %%
train_df.duplicated().sum()
test_df.duplicated().sum()

# %%

negative_review = train_df['Text'][train_df['label']=='-1'].to_string()
wordcloud_negative = WordCloud(width = 800, height = 800, 
                               background_color ='white',
                               min_font_size = 10).generate(negative_review)

positive_review = train_df['Text'][train_df['label']=='1'].to_string()
wordcloud_positive = WordCloud(width = 800, height = 800, 
                               background_color ='white',
                               min_font_size = 10).generate(positive_review)

neutral_review = train_df['Text'][train_df['label']=='0'].to_string()
wordcloud_neutral = WordCloud(width = 800, height = 800, 
                               background_color ='white',
                               min_font_size = 10).generate(neutral_review)
 
# Plotting the WordCloud images                     
plt.figure(figsize=(14, 6), facecolor = None)

plt.subplot(1, 3, 1)
plt.imshow(wordcloud_negative)
plt.axis("off")
plt.title('Negative reviews', fontdict={'fontsize': 20})

plt.subplot(1, 3, 2)
plt.imshow(wordcloud_positive)
plt.axis("off")
plt.title('Positive reviews', fontdict={'fontsize': 20})

plt.subplot(1, 3, 3)
plt.imshow(wordcloud_neutral)
plt.axis("off")
plt.title('neutral reviews', fontdict={'fontsize': 20})


plt.tight_layout() 
plt.show()

# %%
train_df_fe = train_df.copy()
train_df_fe['review_length'] = train_df_fe['Text'].str.len()
train_df_fe['num_words'] = train_df_fe['Text'].apply(lambda x: len(x.split()))
train_df_fe.head()

# %%
plt.figure(figsize=(8, 8))
features = ['review_length','num_words']
for i in range(len(features)):
    plt.subplot(2, 1, i+1)
    sns.distplot(train_df_fe[train_df_fe.label=='1'][features[i]], label = 'Positive')
    sns.distplot(train_df_fe[train_df_fe.label=='-1'][features[i]], label = 'Negative')
    sns.distplot(train_df_fe[train_df_fe.label=='0'][features[i]], label = 'Netural')
    plt.legend()
plt.tight_layout()
plt.show()

# %%
X = train_df.drop(columns=['label'])
y = train_df['label']
test = test_df
print(X.shape, test.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# %%
# def tokenize_and_clean(text):
#     # Changing case of the text to lower case
#     lowered = text.lower()
    
#     # Cleaning the text
#     cleaned = re.sub('@user', '', lowered)
    
#     # Tokenization
#     tokens = word_tokenize(cleaned)
#     filtered_tokens = [token for token in tokens if re.match(r'\w{1,}', token)]
    
#     # Stemming
#     stemmer = PorterStemmer()
#     stems = [stemmer.stem(token) for token in filtered_tokens]
#     return stems
def tokenize_and_clean(text):
    ''' standardize text to extract words
    '''
    # text to lowercase
    text = text.lower()
    # remove numbers
    text = ''.join([i for i in text if not i.isdigit()]) 
    # remove punctuation
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+') # preserve words and alphanumeric
    text = tokenizer.tokenize(text)
    # remove stopwords
    from nltk.corpus import stopwords
    stop = set(stopwords.words('english'))
    text = [w for w in text if not w in stop] 
    # lemmatization
    from nltk.stem import WordNetLemmatizer 
    lemmatizer = WordNetLemmatizer() 
    text = [lemmatizer.lemmatize(word) for word in text]
    # return clean token
    return(text)

# %%
# BOW Vectorization
# bow_vectorizer = CountVectorizer(tokenizer=tokenize_and_clean, stop_words='english')
# X_train_tweets_bow = bow_vectorizer.fit_transform(X_train['tweet'])
# X_test_tweets_bow = bow_vectorizer.transform(X_test['tweet'])
# print(X_train_tweets_bow.shape, X_test_tweets_bow.shape)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_clean, stop_words='english')
X_train_tweets_tfidf = tfidf_vectorizer.fit_transform(X_train['Text'])
X_test_tweets_tfidf = tfidf_vectorizer.transform(X_test['Text'])
print(X_train_tweets_tfidf.shape, X_test_tweets_tfidf.shape)

# TF-IDF Vectorization on full training data
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_clean, stop_words='english')
X_tweets_tfidf = tfidf_vectorizer.fit_transform(X['Text'])
test_tweets_tfidf = tfidf_vectorizer.transform(test['Text'])
print(X_tweets_tfidf.shape, test_tweets_tfidf.shape)

# %%
plt.pie(y_train.value_counts(), 
        labels=['Label 1 (Positive Reviews)', 'Label -1 (Negative Reviews)' , 'Label 0 (Neutral Reviews)'], 
        autopct='%0.1f%%')
plt.axis('equal')
plt.show()

# %%
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train_tweets_tfidf, y_train.values)
print(X_train_smote.shape, y_train_smote.shape)

# SMOTE on full training data
smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X_tweets_tfidf, y.values)
print(X_smote.shape, y_smote.shape)

# Class Imbalance Check
plt.pie(pd.value_counts(y_train_smote), 
        labels=['Label 1 (Positive Reviews)', 'Label -1 (Negative Reviews)' , 'Label 0 (Neutral Reviews)'],  
        autopct='%0.1f%%')
plt.axis('equal')
plt.show()

# %%
from sklearn.metrics import f1_score
def training_scores(y_act, y_pred):
    acc = round(accuracy_score(y_act, y_pred), 3)
    f1 = round(f1_score(y_act, y_pred, average='weighted'), 3)
    print(f'Training Scores: Accuracy={acc}, F1-Score={f1}')
    
def validation_scores(y_act, y_pred):
    acc = round(accuracy_score(y_act, y_pred), 3)
    f1 = round(f1_score(y_act, y_pred, average='weighted'), 3)
    print(f'Validation Scores: Accuracy={acc}, F1-Score={f1}')

# %%
plt.figure(figsize=(8, 8))
features = ['review_length','num_words']
for i in range(len(features)):
    plt.subplot(2, 1, i+1)
    sns.distplot(train_df_fe[train_df_fe.label=='1'][features[i]], label = 'Positive')
    sns.distplot(train_df_fe[train_df_fe.label=='-1'][features[i]], label = 'Negative')
    sns.distplot(train_df_fe[train_df_fe.label=='0'][features[i]], label = 'Netural')
    plt.legend()
plt.tight_layout()
plt.show()

# %%

negative_review = train_df['Text'][train_df['label']=='-1'].to_string()
wordcloud_negative = WordCloud(width = 800, height = 800, 
                               background_color ='white',
                               min_font_size = 10).generate(negative_review)

positive_review = train_df['Text'][train_df['label']=='1'].to_string()
wordcloud_positive = WordCloud(width = 800, height = 800, 
                               background_color ='white',
                               min_font_size = 10).generate(positive_review)

neutral_review = train_df['Text'][train_df['label']=='0'].to_string()
wordcloud_neutral = WordCloud(width = 800, height = 800, 
                               background_color ='white',
                               min_font_size = 10).generate(neutral_review)
 
# Plotting the WordCloud images                     
plt.figure(figsize=(14, 6), facecolor = None)

plt.subplot(1, 3, 1)
plt.imshow(wordcloud_negative)
plt.axis("off")
plt.title('Negative reviews', fontdict={'fontsize': 20})

plt.subplot(1, 3, 2)
plt.imshow(wordcloud_positive)
plt.axis("off")
plt.title('Positive reviews', fontdict={'fontsize': 20})

plt.subplot(1, 3, 3)
plt.imshow(wordcloud_neutral)
plt.axis("off")
plt.title('neutral reviews', fontdict={'fontsize': 20})


plt.tight_layout() 
plt.show()

# %%
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

# Assuming you have defined training_scores and validation_scores functions
# Adjust these functions based on your specific requirements

def training_scores(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f'Training Scores: Accuracy={accuracy:.3f}, F1-Score={f1:.3f}')

def validation_scores(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f'Validation Scores: Accuracy={accuracy:.3f}, F1-Score={f1:.3f}')

# Convert string labels to integers
class_mapping = {'-1': 0, '0': 1, '1': 2}
y_train_smote_int = np.vectorize(class_mapping.get)(y_train_smote).astype(int)
y_test_int = np.vectorize(class_mapping.get)(y_test).astype(int)

# Fit the XGBoost model
xgb = XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss')
xgb.fit(X_train_smote, y_train_smote_int)

# Predictions
y_train_pred = xgb.predict(X_train_smote)
y_test_pred = xgb.predict(X_test_tweets_tfidf)

# Evaluate the model
training_scores(y_train_smote_int, y_train_pred)
validation_scores(y_test_int, y_test_pred)


# %%
import joblib
joblib.dump(xgb, 'XGB_Classifier_model.pkl')

# %%
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Assuming xgb is your trained XGBoost model

# Convert string labels to integers
class_mapping = {'-1': 0, '0': 1, '1': 2}
y_train_smote_int = np.vectorize(class_mapping.get)(y_train_smote).astype(int)

# Fit the XGBoost model with multi:softprob objective
xgb = XGBClassifier(objective='multi:softprob', eval_metric='logloss')
xgb.fit(X_train_smote, y_train_smote_int)

# Predict class probabilities for each class
y_train_pred_probs = xgb.predict_proba(X_train_smote)

# Compute ROC curve and AUC for each class
plt.figure(figsize=(8, 6))
for i in range(len(xgb.classes_)):
    fpr, tpr, _ = roc_curve(y_train_smote_int == xgb.classes_[i], y_train_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {xgb.classes_[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curves for each class')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# %%
from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB()
mnb.fit(X_train_smote,y_train_smote)
y_train_pred = mnb.predict(X_train_smote)
y_test_pred = mnb.predict(X_test_tweets_tfidf)
training_scores(y_train_smote, y_train_pred)
validation_scores(y_test,y_test_pred)

# %%
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Assuming lr is your trained Logistic Regression model

# Fit the Logistic Regression model
mnb.fit(X_train_smote, y_train_smote)

# Predict probabilities for each class
y_train_probs = mnb.predict_proba(X_train_smote)

# Compute ROC curve and AUC for each class
plt.figure(figsize=(8, 6))
for i in range(len(mnb.classes_)):
    fpr, tpr, _ = roc_curve(y_train_smote == mnb.classes_[i], y_train_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {mnb.classes_[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curves for each class')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

lr=LogisticRegression()
lr.fit(X_train_smote,y_train_smote)
y_train_pred = lr.predict(X_train_smote)
y_test_pred = lr.predict(X_test_tweets_tfidf)
training_scores(y_train_smote,y_train_pred)
validation_scores(y_test,y_test_pred)

# %%
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Assuming lr is your trained Logistic Regression model

# Fit the Logistic Regression model
lr.fit(X_train_smote, y_train_smote)

# Predict probabilities for each class
y_train_probs = lr.predict_proba(X_train_smote)

# Compute ROC curve and AUC for each class
plt.figure(figsize=(8, 6))
for i in range(len(lr.classes_)):
    fpr, tpr, _ = roc_curve(y_train_smote == lr.classes_[i], y_train_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {lr.classes_[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curves for each class')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# %%
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train_smote,y_train_smote)
y_train_pred=rf.predict(X_train_smote)
y_test_pred=rf.predict(X_test_tweets_tfidf)
training_scores(y_train_smote,y_train_pred)
validation_scores(y_test,y_test_pred)

# %%
#hyperparameter tuning
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(criterion='entropy', max_samples=0.8,min_samples_split=10,random_state=0)
rf.fit(X_train_smote,y_train_smote)
y_train_pred=rf.predict(X_train_smote)
y_test_pred = rf.predict(X_train_tweets_tfidf)
training_scores(y_train_smote,y_train_pred)
validation_scores(y_test,y_test_pred)

# %%
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Assuming lr is your trained Logistic Regression model

# Fit the Logistic Regression model
rf.fit(X_train_smote, y_train_smote)

# Predict probabilities for each class
y_train_probs = rf.predict_proba(X_train_smote)

# Compute ROC curve and AUC for each class
plt.figure(figsize=(8, 6))
for i in range(len(rf.classes_)):
    fpr, tpr, _ = roc_curve(y_train_smote == rf.classes_[i], y_train_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {rf.classes_[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curves for each class')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


