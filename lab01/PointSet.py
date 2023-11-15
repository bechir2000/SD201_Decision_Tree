from typing import List, Tuple

from enum import Enum
import numpy as np

class FeaturesTypes(Enum):
    """Enumerate possible features types"""
    BOOLEAN=0
    CLASSES=1
    REAL=2

class PointSet:

    def __init__(self, features: List[List[float]], labels: List[bool], types: List[FeaturesTypes]):
        self.types = types
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.best_ind = None
        self.split_value = None
    
    def get_gini(self):
        # calculate the gini of the PointSet
        labels = self.labels
        n = len(labels)
        # Calculate the proportion of positive and negative labels
        p_positive = labels.sum() / n
        p_negative = 1 - p_positive
    
        # Calculate the Gini impurity
        gini = 1 - (p_positive**2 + p_negative**2)
    
        return gini
        
    def calc_feat_index(self,feat_ind: int, split_value: float = 1.0):
        """Distinguish 2 cases : BOOLEAN and CLASSES
        if CLASSES, then feature is True if equal to split_value"""
        feature_true = []
        feature_false = []
        label_true, label_false = [],[]
        types = self.types[feat_ind]
        for j in range(len(self.features)):
            if types == FeaturesTypes.BOOLEAN:
                    if self.features[j][feat_ind]:
                        feature_true.append(self.features[j])
                        label_true.append(self.labels[j])
                    else:   
                        feature_false.append(self.features[j])
                        label_false.append(self.labels[j])
            elif types == FeaturesTypes.CLASSES:
                    if self.features[j][feat_ind] == split_value: 
                        feature_true.append(self.features[j])
                        label_true.append(self.labels[j])
                    else:
                        feature_false.append(self.features[j])
                        label_false.append(self.labels[j])
        return feature_true, feature_false, label_true, label_false
    
    
    def calc_gini_split(self, feat_ind: int, split_value: float = 1.0):
        """Computes the Gini_split score of points after splitting
        them along a feature
        """
        feature_true, feature_false, label_true, label_false = self.calc_feat_index(feat_ind, split_value)
        n=len(self.features)
        if (len(feature_true) == 0 or len(feature_false) == 0):
           return None
        
        ## calculate gini for True
        gini_true = PointSet(feature_true,label_true,[]).get_gini()
        
        ##calculate gini for False
        gini_false = PointSet(feature_false,label_false,[]).get_gini()
        
        ## calculate gini_split
        gini_split = (len(feature_true)/n)*gini_true + (len(feature_false)/n)*gini_false
        return gini_split
         
                  
    def get_best_gain(self):
        """Compute the feature along which splitting provides the best gain
        """   
        max_gini = 0.0
        best_ind = None
        best_split_value = None
        gini = self.get_gini()
        
        # split the set along each feature and calculate gini gain
        # we distinguish 2 cases: BOOLEAN and CLASSES
        for feat in range(len(self.features[0])):
            split_value = None
            types = self.types[feat]
            # 1st case : BOOLEAN
            if types == FeaturesTypes.BOOLEAN:
                split_value = 1.0
                gini_split = self.calc_gini_split(feat, split_value)
                if gini_split == None:
                    continue 
                gini_gain = gini - gini_split # According to slide 32
                if (gini_gain > max_gini):
                    max_gini = gini_gain
                    best_ind = feat
                    best_split_value = split_value
                    
            # 2st case : CLASSES
            elif types == FeaturesTypes.CLASSES:
                feature_values = list()  # possible values of a feature
                for i in range(len(self.features)):
                    if self.features[i][feat] not in feature_values:
                        feature_values.append(self.features[i][feat])
                
                for split_value in feature_values:
                    gini_split = self.calc_gini_split(feat, split_value)
                    if gini_split == None:
                        continue
                    gini_gain = gini - gini_split
                    if (gini_gain > max_gini):
                        max_gini = gini_gain
                        best_ind = feat
                        best_split_value = split_value
        if (max_gini == 0.0):
            return (None, None, None)
        
        self.best_ind = best_ind
        self.split_value = best_split_value
        return best_ind, max_gini
            

    def split_with_best_gain(self):
        self.get_best_gain()
        if self.best_ind == None:
            return (None, None, None, None, None, None)
        
        true_class_features, false_class_features, true_class_labels, false_class_labels = self.calc_feat_index(self.best_ind,self.split_value)
        return (self.best_ind, self.split_value, true_class_features, true_class_labels, false_class_features, false_class_labels)
        
