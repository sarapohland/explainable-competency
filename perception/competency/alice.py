import os
import json
import numpy as np

import sklearn
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from perception.network.model import NeuralNet

# Accurate Layerwise Interpretable Compentence Estimation
class Alice:

    def __init__(self, train_loader, model_dir):

        # Set default threshold
        self.thresh = 0.99

        # Load trained model
        with open(os.path.join(model_dir, 'layers.json')) as file:
            layer_args = json.load(file)
        self.model = NeuralNet(layer_args)
        self.model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')))
        self.model.eval()

        # Get features and predictions of trained model
        all_lbls, all_feats, all_preds = [], [], []
        for X, y in train_loader:
            with torch.set_grad_enabled(False):
                features = self.model.get_feature_vector(X).detach()
                outputs  = self.model(X)
                _, preds = torch.max(outputs, 1)
                all_lbls.append(y)
                all_feats.append(features)
                all_preds.append(preds[:,None])
        data  = torch.vstack(all_feats)
        preds  = torch.vstack(all_preds)
        labels = torch.vstack(all_lbls)

        # Perform PCA on training data
        n_components = 0.99
        self.pca = PCA(n_components=n_components)
        data = self.pca.fit_transform(data)

        # Initialize GMM distributions
        self.mu = []
        self.prec = []
        self.mahalanobis_classifiers = []
        self.max_p_d = False
        
        # Fit Gaussian distributions for each class
        competence = (preds==labels)
        comp_inds = np.where(competence)[0]
        self.n_classes = len(np.unique(labels))
        for c in range(self.n_classes):
            print("Fitting Gaussians {}/{}...".format(c+1, self.n_classes))
            label_inds = np.where(labels == c)[0]
            mask = np.intersect1d(comp_inds, label_inds, assume_unique=True)
            
            # Skip class if there's fewer than two data points correctly predicted
            if len(mask) < 2:
                self.mu.append(None)
                self.prec.append(None)
                self.mahalanobis_classifiers.append(None)
                continue
            
            # Compute the mean, covariance, and precision matrices
            mu = np.mean(data[mask], axis=0)
            cov = np.cov(data[mask], rowvar=0)
            prec = np.linalg.pinv(cov)
            self.mu.append(mu)
            self.prec.append(prec)
            
            # Compute Mahalanobis distances
            print("Getting mahalanobis distances...")
            maha_distances = self._get_maha_distances(mu, prec, data)

            # Create logistic regression model from distances
            ood_lbls = np.zeros_like(maha_distances)
            ood_lbls[mask] = 1
            self.mahalanobis_classifiers.append(self._fit_maha_logreg(maha_distances, ood_lbls))
        
        # Train transfer classifier for class probabilities
        print("Fitting transfer classifier...")
        self.calibrated_transfer = self._fit_calibrated_transfer(data, labels)
        
        print("Done initializing ALICE model!")
        
    # Compute Mahalanobis distances
    def _get_maha_distances(self, mu, prec, data):
        if self.mu is None:
            return float('inf')
        dist = sklearn.metrics.DistanceMetric.get_metric('mahalanobis', VI=prec)
        m_distances = dist.pairwise(data, mu.reshape(1, -1))
        return m_distances

    # Fit logistic regression model (to convert distances to probabilities)
    def _fit_maha_logreg(self, maha_distances, labels, balance=True):
        if balance:
            model = sklearn.linear_model.LogisticRegressionCV(max_iter=4000, class_weight='balanced')
        else:
            model = sklearn.linear_model.LogisticRegressionCV(max_iter=4000)
        return model.fit(maha_distances, labels)
        
    # Fit calibrated transfer classifier model to approximate p(c|x,D)
    def _fit_calibrated_transfer(self, data, labels):
        model = sklearn.linear_model.LogisticRegressionCV(max_iter=4000)
        return model.fit(data, labels)

    # Get calibrated prediction for class c
    def _get_p_c(self, c, calibrated_preds):
        return calibrated_preds[:, c]

    # Estimate the competency score using ALICE
    def comp_score(self, inputs, outputs):
        # Get features of trained model
        if len(inputs.size()) > 2:
            inputs = self.model.get_feature_vector(inputs)

        alice_class = []
        logreg_class = []
 
        # Perform principal component analysis
        V = torch.from_numpy(self.pca.components_)
        m = torch.from_numpy(self.pca.mean_)
        inputs = (inputs - m) @ V.T

        # Get calibrated predictions of inputs
        W = torch.from_numpy(self.calibrated_transfer.coef_)
        b = torch.from_numpy(self.calibrated_transfer.intercept_)
        calibrated_preds = F.softmax(inputs @ W.T + b, dim=1)

        # Compute Mahalanobs distance predictions
        dist_classes = []
        for c in range(self.n_classes):
            m = torch.from_numpy(self.mu[c])
            Sinv = torch.from_numpy(self.prec[c])
            w = torch.from_numpy(self.mahalanobis_classifiers[c].coef_)
            b = torch.from_numpy(self.mahalanobis_classifiers[c].intercept_)
            d = torch.sqrt((inputs - m) @ Sinv @ (inputs - m).T)
            p_d = F.sigmoid(w * d + b)
            dist_classes.append(p_d)
        dist_classes = torch.stack(dist_classes)
        max_p_d = torch.max(dist_classes, axis=0)
        
        # Compute each term within sum of ALICE score
        for c in range(self.n_classes):
            # Estimate the prob. that inputs are in-distribution (ID)
            m = torch.from_numpy(self.mu[c])
            Sinv = torch.from_numpy(self.prec[c])
            w = torch.from_numpy(self.mahalanobis_classifiers[c].coef_)
            b = torch.from_numpy(self.mahalanobis_classifiers[c].intercept_)
            D = torch.sqrt((inputs - m) @ Sinv @ torch.swapaxes((inputs - m), -1, -2))
            d = torch.diagonal(D, dim1=-1, dim2=-2)
            p_d = F.sigmoid(w * d + b)

            # Estimate prob. that inputs are in class c, given they are ID
            p_c = self._get_p_c(c, calibrated_preds)

            # Get indicator for whether prediction = class c
            ind = torch.zeros((inputs.size()[0]))
            corr_inds = torch.where(torch.argmax(outputs, axis=1) == c)[0]
            ind[corr_inds] = 1

            # Compute "sub-ALICE score" for each class
            a_c = p_d * p_c * ind
            if self.max_p_d:
                a_c = max_p_d * p_c * ind
            
            # Save results for given class
            alice_class.append(a_c)

        # Compute ALICE and logistic regression scores
        alice_class = torch.stack(alice_class)
        alice_scores = torch.sum(alice_class, axis=0).flatten()
        return alice_scores
    
    # Set the threshold for model competency 
    def set_threshold(self, thresh):
        self.thresh = thresh

    # Determine if model is competent
    def comp_dec(self, scores, preds=None):
        return (scores > self.thresh)