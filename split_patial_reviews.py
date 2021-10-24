import json
import string
import pandas as pd
# import spacy
# import nltk

count = 0
reviews=[]
stars=[]
business_id=[]
# nltk.download('punkt')
# en = spacy.load('en_core_web_sm')
# excluded_words_list = en.Defaults.stop_words
# for p in string.punctuation:
#     excluded_words_list.add(p)
# excluded_words_list.add('\'s')
# excluded_words_list.add('n\'t')
# excluded_words_list.add("''")
# excluded_words_list.add("``")

with open('/Users/enity/Downloads/text_group_work/yelp_dataset/yelp_academic_dataset_review.json', encoding='utf-8') as f:
    for line in f:
        line_contents = json.loads(line)
        reviews.append(line_contents['text'])
        stars.append(int(line_contents['stars']))
        # business_id.append(line_contents['business_id'])
        count+=1
        if count == 400000:
            break
# print("open file end")
# final_reviews=[]
# for r in reviews:
#     tokens = nltk.word_tokenize(r)
#     after_tokens = [word for word in tokens if word not in excluded_words_list]
#     final_reviews.append(after_tokens)

# df = pd.DataFrame({"text":reviews,"stars":stars,"business_id":business_id})

df = pd.DataFrame({"text":reviews,"stars":stars})
df = df.loc[300000:400000,:]
df = df.loc[df['stars'] != 3]
df['sentiment']=df.apply(lambda x: 1 if x.stars>=4 else 0,axis=1)
df = df.drop('stars',axis=1)
print(df.shape)
df.to_csv("reviews_train_dataset.csv",index=False)
print('end')
# df_low_stars=df.loc[df['stars']==1,:]

# business_list=set()
# for index,row in df_low_stars.iterrows():
#     business_list.add(row['business_id'])
#     if len(business_list) == 50:
#         break

# print(len(business_list))

# df_result = pd.DataFrame(columns=['text','stars','business_id'])
# for b_id in business_list:
#     result = df.loc[(df["business_id"] == b_id) & (df['stars'] == 1), :]
#     df_result=df_result.append(result.iloc[0])
# df_result.to_csv("random_result.csv",index=False)


