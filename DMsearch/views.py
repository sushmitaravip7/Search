
from django.views.generic import TemplateView
from django.core import serializers
import nltk
import ssl
from nltk.corpus import stopwords

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import re


class SushDM(TemplateView):
        template_name = "test.html"

        def post(self, request, *args, **kwargs):
                #Get Query from User
                query1 =request.POST.get("firstname",None)
                #Convert Query to lower case to avoid case sensitive issues
                query = query1.lower()

                nltk.download('stopwords')

                #query = "love story"
                
                #Stop words
                stop_words = set(stopwords.words('english'))
                
                #read from csv
                df = pd.read_csv('example.csv',low_memory=False,dtype={"bookID": int, "title": str, "author":str, "rating": float, "ratingsCount": int, "reviewsCount": int,
                                                 "reviewerName": str, "reviewerRatings": int, "review": str})

                df.dropna(inplace = True)

                #Count the number of times the word has appeared in each document
                word_vec = df["review"].str.lower().str.replace(r'[^\w\s]+', ' ').apply(str.split).apply(lambda x: [item for item in x if item not in stop_words]).apply(pd.value_counts).fillna(0)

                #Count the number of times the word has appeared in the query
                query_df = pd.Series(0, index=word_vec.columns)
                
                found=False
                for word in query.split():
                    try:
                        query_df[word] +=1
                        found=True
                    except:
                        pass
                if found==False :
                        context = self.get_context_data()
                        context["result"]  = "No Match Found"
                        return super(TemplateView, self).render_to_response(context)
                                                   
                
                # Compute term frequencies for the documents and the query
                tf = word_vec.divide(np.sum(word_vec, axis=1), axis=0)
                query_tf = query_df.apply(lambda x: x / np.sum(query_df))


                # Compute inverse document frequencies
                idf = 1 + np.log10(len(word_vec) / word_vec[word_vec > 0].count())

                # Compute TF-IDF vectors
                tfidf = np.multiply(tf, idf.to_frame().T)
                query_tfidf = np.multiply(query_tf, idf.T)


                #Cosine Similarity
                cosine_similarities = cosine_similarity(tfidf, query_tfidf.to_frame().T).flatten()
                #Add to Datafame
                df['Similarity'] = cosine_similarities
                #rank Documents
                df.sort_values("Similarity", inplace=True, ascending=False)
                #Top 5 
                display=df.head(n=5)

                #print(display)
                display1=display[['title', 'author', 'review']]
                
               
                
                

                context = self.get_context_data()

                context["result"]  = display1
                context["searchWord"] = query1
                return super(TemplateView, self).render_to_response(context)
                

                                    
