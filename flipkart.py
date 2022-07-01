from multiprocessing.sharedctypes import Value
import pandas as pd ; import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
import matplotlib.pyplot as plt

nltk.download('stopwords',quiet=True)
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
stopword=set(stopwords.words('english'))

data = pd.read_csv("sample_data.csv")
REVIEW="review"
RATING="rating"

# visual analysis of reviews given
text = " ".join(data[REVIEW].to_numpy())
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# analysis of ratings
rat_num = data[RATING].to_numpy()
neg_probab = 0
for item in rat_num:
    if item<=2: neg_probab+=1
neg_probab/=sum(rat_num)

nltk.download('vader_lexicon',quiet=True)
sentiments = SentimentIntensityAnalyzer()
pos,neg,neu = 0,0,0
def sps(x): return sentiments.polarity_scores(x)
count = -1 ; neg_fb=[] ; accuracy = 0

# analysis of reviews
for item in data[REVIEW]:
    count+=1
    sentences = item.split(".")
    n = len(sentences) ; x,y,z=0,0,0
    for line in sentences:
        sent_val = sps(line)
        x,y,z = x+sent_val["pos"],y+sent_val["neg"],z+sent_val["neu"]
        if sent_val["neg"]>max(sent_val["pos"],sent_val["neu"]):
            if "READ" and "READ MORE" not in line:  neg_fb.append(line)
    
    if (y+z>x and rat_num[count]<=2) or (x+z>y and rat_num[count]>=4):
        accuracy+=1    
    temp = x+y+z
    try: pos+=x/(n*temp) ; neg+=y/(n*temp) ; neu+=z/(n*temp)
    except ZeroDivisionError: pass

neg_val = neg/(pos+neg+neu)
goodFb1 = round(100*(1-neg_val),2) ; goodFb2 = round(100*(1-neg_probab),2)
accuracy*=100/len(rat_num)
print("Percentage of good feedback from reviews: ",goodFb1)
print("Percentage of good feedback from ratings: ",goodFb2)
print("Accuracy of sentiment analysis: ",accuracy)
print("\nA few points of criticism we observed:")
from random import randint
n = len(neg_fb) ; count=1
while count<=5:
    i = randint(0,n-1)
    temp = sentiments.polarity_scores(neg_fb[i])
    if temp["pos"]>=temp["neg"]: continue
    else: print(neg_fb[i]) ; count+=1
