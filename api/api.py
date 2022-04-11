from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
import pickle
import numpy as np
import json

import os
import sys
from pathlib import Path
import tensorflow as tf

from utilities.training.loader import load_from_path
from utilities.predict.cluster import predict_cluster

from utilities.config import ROOT_DIR
from utilities.training.loader import load_images, load_model
from utilities.training.features import img_convert, dim_reduce
from utilities.training.clustering import k_means_clustering

import random

path = load_from_path()

CWD = os.getcwd()
sys.setrecursionlimit(5000)

class Train(Resource):
    def get(self):
        # Load all the Images and shuffle them
        image_key, image_list, num_images = load_images(path)
        for i in range (3):
            random.shuffle(image_list)

        # Load the inception model
        model_inception = load_model()

        # Obtain the feature vector form of all the images 
        features, filenames = img_convert(image_list, model_inception)

        # Reduce the Dimensions of the features
        pca, new_features = dim_reduce(features, 1)

        # Performing K-Means Clustering to identify suitable number of Clusters
        ans, clusters, labels, kmeans = k_means_clustering(new_features, filenames, num_images)

        '''
            After training and creating clusters we will dump all this data into a pickle file 
            which we can use later on for predictions.
        '''

        # Dumping the Model into a pickle file
        pickle.dump(kmeans, open(str(CWD + "/pickle dumps/clustering_model.pkl"), 'wb'))

        # Dumping the pca fit to transform the test image
        pickle.dump(pca, open(str(CWD + "/pickle dumps/pca_fit.pkl"), 'wb'))

        # Dumping the pretrained model which will extract features for a given image.
        # pickle.dump(model_inception, open(str(CWD + "/pickle dumps/model_inception.pkl"), 'wb'))
        model_inception.save(str(CWD + '/pickle dumps/model_inception.h5'))

        # Dumping the Clusters List
        pickle.dump(clusters, open(str(CWD + "/pickle dumps/clusters.pkl"), 'wb'))
    

api.add_resource(Train, '/train')

class author(Resource):
    def post(self):
        # Loading all the pickle files
        model_inception = tf.keras.models.load_model(str(CWD + '/pickle dumps/model_inception.h5'))
        pca_model = pickle.load(open(str(CWD + '/pickle dumps/pca_fit.pkl'), 'rb'))
        clustering_model = pickle.load(open(str(CWD + '/pickle dumps/clustering_model.pkl'), 'rb'))
        clusters = pickle.load(open(str(CWD + '/pickle dumps/clusters.pkl'), 'rb'))

        final = {}
        def suggest_authors (clustering_model, pca_model, model, clusters, path):
            authors = {}
            for root, dir, files in os.walk(path):
                for img in files:
                    if '.jpg' in img:
                        img_path = os.path.join(root, img)
                        new_cluster = predict_cluster(clustering_model, pca_model, img_path, model)
                        for i in clusters[new_cluster[0]]:
                            p = Path(i).parents[0]
                            name = os.path.basename(p)
                            if name in authors:
                                authors[name] += 1
                            else:
                                authors[name] = 1
            return authors

        author_list = suggest_authors (clustering_model, pca_model, model_inception, clusters, path)

        return jsonify(author_list)

api.add_resource(author, '/predict')


app = Flask(__name__)
api = Api(app)


if __name__ == '__main__':
    app.run(debug=True)