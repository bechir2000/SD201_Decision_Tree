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
        if types == FeaturesTypes.BOOLEAN:
            for j in range(len(self.features)):
                if self.features[j][feat_ind]:
                    feature_true.append(self.features[j])
                    label_true.append(self.labels[j])
                else:   
                    feature_false.append(self.features[j])
                    label_false.append(self.labels[j])
        elif types == FeaturesTypes.CLASSES:
                if split_value == None:
                    raise ValueError("split_value : None")
                for j in range(len(self.features)):
                    if self.features[j][feat_ind] == split_value: 
                        feature_true.append(self.features[j])
                        label_true.append(self.labels[j])
                    else:
                        feature_false.append(self.features[j])
                        label_false.append(self.labels[j])
        elif types == FeaturesTypes.REAL:
                if split_value == None:
                    raise ValueError("split_value : None")
                for j in range(len(self.features)):
                    if self.features[j][feat_ind] < split_value:
                        feature_true.append(self.features[j])
                        label_true.append(self.labels[j])
                    else:
                        feature_false.append(self.features[j])
                        label_false.append(self.labels[j])                    
        return feature_true, feature_false, label_true, label_false

    def calc_gini_split(self, feat_ind: int, split_value: float = None, min_split_points: int = 1):
        
        feature_true, feature_false, label_true, label_false = self.calc_feat_index(feat_ind, split_value)         
        n=len(self.features)
       
        if (len(feature_true) == 0 or len(feature_false) == 0):
           return None
       
        # constraint violated
        if (len(feature_true) < min_split_points or len(feature_false) < min_split_points):
            return None
        
        ## calculate gini for True
        gini_true = PointSet(feature_true,label_true,[]).get_gini()
        
        ##calculate gini for False
        gini_false = PointSet(feature_false,label_false,[]).get_gini()
        
        ## calculate gini_split
        gini_split = (len(feature_true)/n)*gini_true + (len(feature_false)/n)*gini_false
        return gini_split
    
    
    def get_best_gain(self,min_split_points : int = 1):
        
        max_gini_gain = 0.0
        best_ind = None
        best_split_value = None
        gini = self.get_gini()
        
        ## split the set along each feature (each value if type is not bool) and calculate gini gain
        for feat in range(len(self.features[0])):
            split_value = None
            types = self.types[feat]
            # 1st case : BOOLEAN
            if types == FeaturesTypes.BOOLEAN:
                gini_split = self.calc_gini_split(feat, split_value, min_split_points)
                if gini_split == None:
                    continue 
                gini_gain = gini - gini_split
                if (gini_gain > max_gini_gain):
                    max_gini_gain = gini_gain
                    best_ind = feat
                    best_split_value = split_value  
            
            # 2st case : CLASSES
            elif types == FeaturesTypes.CLASSES:
                feature_values = list()
                for i in range(len(self.features)):
                    if self.features[i][feat] not in feature_values:
                        feature_values.append(self.features[i][feat])
                for split_value in feature_values:
                    gini_split = self.calc_gini_split(feat, split_value, min_split_points)
                    if gini_split == None:
                        continue
                    gini_gain = gini - gini_split
                    if (gini_gain > max_gini_gain):
                        max_gini_gain = gini_gain
                        best_ind = feat
                        best_split_value = split_value
            
            # 3rd case : REAL          
            elif types == FeaturesTypes.REAL:
                feature_values = list() # 
                split_values = [] # store all possible split values of a feature
                for i in range(len(self.features)):
                    if self.features[i][feat] not in feature_values:
                        feature_values.append(self.features[i][feat])
                feature_values.sort()
                for i in range(len(feature_values)-1):
                    split_values.append((feature_values[i]+feature_values[i+1])/2)
                for split_value in split_values:
                    gini_split = self.calc_gini_split(feat, split_value, min_split_points)
                    if gini_split == None:
                        continue
                    gini_gain = gini - gini_split
                    if (gini_gain > max_gini_gain):
                        max_gini_gain = gini_gain
                        best_ind = feat
                        best_split_value = split_value
        
        if (max_gini_gain == 0.0):
            return None, None, None
        
        self.best_ind = best_ind
        self.split_value = best_split_value
        return best_ind, max_gini_gain
            
       
    def get_best_threshold(self):
    
        if self.best_ind == None:
            raise ValueError("get_best_gain() must be called first")
        return self.split_value     

    def split_with_best_gain(self, min_split_points : int = 1):
        
        # set self.best_ind and self.split_value
        self.get_best_gain(min_split_points)
        # if no split can reduces gini, return (None, None, None, None, None)
        if self.best_ind == None:
            return (None, None, None, None)
        
        true_class_features, false_class_features, true_class_labels, false_class_labels = self.calc_feat_index(self.best_ind,self.split_value)

        return (self.best_ind, self.split_value, true_class_features, true_class_labels, false_class_features, false_class_labels)
        
