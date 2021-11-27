import streamlit as st
from googlesearch import search
from newspaper import Article
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt')

st.title('Fake News Identification using Web scraping & NLP')
inp=st.text_input("Enter News: ")
time=st.selectbox('Select Time:',('Last Hour', 'Last Day', 'Last Week', 'Last Month', 'Anytime'))
btn=st.button('Check')
ts={'Last Hour':"qdr:h",'Last Day':"qdr:d",'Last Week':'qdr:w','Last Month':"qdr:m",'Anytime':"qdr:none"}

if btn:
    query = "NEWS "+inp

    max_a=0
    max_b=0
    url='none'
    link='none'
    for url in search(query, num=15, tbs=ts[time], start=1, stop= 15, pause=3): 
    #tbs (str) – Time limits (i.e “qdr:h” => last hour, “qdr:d” => last 24 hours, “qdr:m” => last month)
        #print(url)
        
        try:
            toi_article = Article(url, language="en")
            toi_article.download()
            toi_article.parse()
            toi_article.nlp()
        except:
            print('***********Exception*****************')
            continue
        
        summary = toi_article.summary
        corpus = [summary, inp]
        vect = TfidfVectorizer(min_df=1, stop_words="english")
        tfidf = vect.fit_transform(corpus)
        pairwise_similarity = tfidf * tfidf.T
        a = pairwise_similarity.toarray()

        title = toi_article.title
        corpus = [title, inp]
        vect = TfidfVectorizer(min_df=1, stop_words="english")
        tfidf = vect.fit_transform(corpus)
        pairwise_similarity = tfidf * tfidf.T
        b = pairwise_similarity.toarray()
        
        max_num = max(max_a,max_b)
        if (a[0][1]>max_a) or (b[0][1]>max_b):
            max_a=a[0][1]
            max_b=b[0][1]
            link=url
    res=int(round(max(max_a,max_b),2)*100)
    if res>70:
        st.write(f'''Most Relevant News: {link}''')
        st.success(f'''Genuinity: {res}%''')
    elif res>20:
        st.write(f'''Most Relevant News: {link}''')
        st.success(f'''Genuinity: {res+30}%''')
    else:
        st.write(f'''It Might be a Fake News''')
        st.error(f'''Genuinity: {res+5}%''')
    btn=None
