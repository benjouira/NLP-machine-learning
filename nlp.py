@author: rabi3
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import *


df = pd.read_csv('train.csv', encoding='latin-1')

print('---The basic info of the data---')
print(df.info())
print(df.shape)


print('---The head/Tail of the data---')
print(df.head())
print('*******************************')
print(df.tail())


df['Sentiment'].plot(kind='hist')
plt.show()

#remove stop word

short_data = df.head()
stop = stopwords.words("english")
print(short_data['SentimentText'])
print('------Remove stop word-----')
short_data['Step1_SentimentText']=short_data['SentimentText'].apply(lambda x : ' '.join([word for word in x.split() if word not in (stop)]))
print(short_data['Step1_SentimentText'])


#Step 2 replace special char and replace abbreviations 
import csv,re
print('---------------------------')

def translator(user_string):
    user_string = user_string.split(" ")
    j = 0
    for _str in user_string:
        # File path which consists of Abbreviations.
        fileName ="slang.txt"
        # File Access mode [Read Mode]
        AccessMode = "r"
        with open(fileName, AccessMode) as myCSVfile:
            # Reading file as CSV with delimiter as "=", so that abbreviation are stored in row[0] and phrases in row[1]
            dataFromFile = csv.reader(myCSVfile, delimiter="=")
            # Removing Special Characters.
            _str = re.sub('[^a-zA-Z0-9-_.]', '', _str)
            for row in dataFromFile:
                # Check if selected word matches short forms[LHS] in text file.
                if _str.upper() == row[0]:
                    # If match found replace it with its appropriate phrase in text file.
                    user_string[j] = row[1]
            myCSVfile.close()
        j = j + 1
    # Replacing commas with spaces for final output.
    return(' '.join(user_string))
    print('===========')
    print('')
while True:
    print("Provide Input below or print exit or EXIT to end script")
    # Getting User String.
    # Sample : user input = "Hi Rishabh meet me asap!"
    user = input()
    # Keep Calling procedure until EXIT or exit keyword is encountered.
    if user.upper() == 'EXIT':
        print("Exiting Script")
        break
    translator(user)
print(short_data['Step1_SentimentText'])
print('------Replace Abbreviations------')

short_data['Step2_SentimentText']=short_data['Step1_SentimentText'].apply(lambda x:translator(x))
print(short_data['Step2_SentimentText'])

print("***********")


#step 3 Stemming
ps=PorterStemmer()
print(short_data['Step2_SentimentText'])
print('-----Stemming-----')
short_data['Step3_SentimentText'] = short_data['Step2_SentimentText'].apply(lambda x : ' '.join([ps.stem(word) for word in x.split() ]))
print(short_data['Step3_SentimentText'])


print("*****************")
#step 4 lemmazation
lmtzr= WordNetLemmatizer()
print(short_data['Step2_SentimentText'])

short_data['Step4_SentimentText']=short_data['Step2_SentimentText'].apply(lambda x : ' '.join([lmtzr.lemmatize(word, 'v') for word in x.split() ]))
print(short_data['Step4_SentimentText'])


#step 5 Parts of Speech Tagging (POS) [New Feature]
print(short_data['Step2_SentimentText'])
print('-----Part of S peech Tagging-----')
short_data['Step5_SentimentText']=short_data['Step2_SentimentText'].apply(lambda x : nltk.pos_tag(nltk.word_tokenize(x)))
print(short_data['Step5_SentimentText'])


#step 6 Capitalization
print(short_data['Step2_SentimentText'])
print('---------Capitalization---------')
short_data['Step6_SentimentText'] = short_data['Step2_SentimentText'].apply( lambda x : ' '.join( [ word.upper() for word in x.split() ]))
print(short_data['Step6_SentimentText'])

