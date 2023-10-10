#!/usr/bin/env python
import click
import pandas as pd
import time
from pymongo import MongoClient
import nltk
from nltk.corpus import stopwords
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
nltk.download('vader_lexicon')
client = MongoClient('localhost', 27017)
nlp = spacy.load('en_core_web_sm')
db = client['frag']
collection = db['dataset_all']
@click.command()
@click.argument('csv_path')
def import_heading(csv_path):
    start_time = time.time()
    try:
        # loading the csv data
        df = pd.read_csv(csv_path)
        data = df.to_dict(orient='records')
        collection.insert_many(data)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{execution_time:.2f}")
    except Exception as e:
        print(f"Error: {str(e)}")
## second task :
def extract_data():
    start_time = time.time()
    cursor = collection.find()
    for document in cursor:
        text = document["headline_text"]
        doc = nlp(text)
        entities = [{ "ent_name" : ent.label_, "ent_text" : ent.text } for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "LOC"]]
        sentiment = sid.polarity_scores(text)
        sentiment_label = "positive" if sentiment["compound"] > 0 else "negative" if sentiment["compound"] < 0 else "neutral"
        collection.update_one({"_id": document["_id"]},{"$set": {"entities": entities, "sentiment": sentiment_label}})
    end_time = time.time()
    execution_time = end_time - start_time
    print(execution_time)
# third task :
from collections import Counter
def get_100_data():
    start_time = time.time()
    documents = collection.find({}, {"entities": 1})
    entity_counter = Counter()
    entity_name_arr = []
    allowed_entity_types = ["PERSON", "ORG", "LOC"]
    for document in documents:
        entities = document.get("entities", [])
        for entity in entities:
            entity_text = entity.get("ent_text")
            entity_type = entity.get("ent_name")
            if entity_type in allowed_entity_types:
                entity_counter[(entity_text, entity_type)] += 1
    top_100_entities = entity_counter.most_common(100)
    for i, (entity, count) in enumerate(top_100_entities, start=1):
        entity_text, entity_type = entity
        entity_name_arr.append(entity_text)
    end_time = time.time()
    execution_time = end_time - start_time
    print(execution_time)
    return entity_name_arr
if __name__ == '__main__':
    import_heading()
    extract_data()
    get_100_data()
    