import os
import random

from utils import text_prepare, load_embeddings, question_to_vec, unpickle_file

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_argmin
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer


class ThreadRanker:
    def __init__(self):
        self.word_embeddings, self.embeddings_dim = load_embeddings('./starspace_embeddings/data/stackoverflow_duplicate.tsv')
        self.thread_embeddings_folder = './data/thread_embeddings_by_tag'

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name[0] + '.pkl')
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)
        
        question_vec = question_to_vec(question, self.word_embeddings, self.embeddings_dim)
        best_thread = pairwise_distances_argmin(np.array([question_vec]), thread_embeddings, metric='cosine')
        
        return thread_ids[best_thread[0]]


class DialogueManager:
    def __init__(self):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = unpickle_file('./model_artifacts/intent_recognizer.pkl')
        self.tfidf_vectorizer = unpickle_file('./model_artifacts/tfidf_vectorizer.pkl')

        self.ANSWER_TEMPLATE = [
            'I think it\'s about %s. This thread might help you: https://stackoverflow.com/questions/%s',
            'Um.. %s .. check this out https://stackoverflow.com/questions/%s',
            'Glad u asked about %s, here you go! https://stackoverflow.com/questions/%s'
        ]

        # Goal-oriented part:
        self.tag_classifier = unpickle_file('./model_artifacts/tag_classifier.pkl')
        self.thread_ranker = ThreadRanker()

    def create_chitchat_bot(self, train=False):
        """Initializes self.chitchat_bot with some conversational model."""

        self.chatbot = ChatBot('Alex')

        if train:
            trainer = ChatterBotCorpusTrainer(self.chatbot)
            trainer.train('chatterbot.corpus.english')
       
    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""
        
        prepared_question = text_prepare(question)
        features = self.tfidf_vectorizer.transform(np.array([prepared_question])) 
        intent = self.intent_recognizer.predict(features) 

        # Chit-chat part:   
        if intent == 'dialogue':       
            response = self.chatbot.get_response(question) 
            return response
        
        # Goal-oriented part:
        else:        
            tag = self.tag_classifier.predict(features)
            thread_id = self.thread_ranker.get_best_thread(prepared_question, tag)
           
            return random.choice(self.ANSWER_TEMPLATE) % (tag[0], thread_id)

