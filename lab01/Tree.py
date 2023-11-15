from typing import List

from PointSet import PointSet, FeaturesTypes
    
class Tree:
    """A decision Tree

    Attributes
    ----------
        points : PointSet
            The training points of the tree
        root : Node
            The root of the tree
        best_ind : int
            The index of the feature along which the points have been split
        split_value : float
            The value of the feature along which the points have been split
            for boolean feature, split_value = 1.0
    """
            
    def __init__(self,
                 features: List[List[float]],
                 labels: List[bool],
                 types: List[FeaturesTypes],
                 h: int = 1):

        self.points = PointSet(features,labels,types)
        best_feat = self.points.get_best_gain()[0]
        if best_feat is None or h == 0:
            # No more splitting possible
            self.left = None
            self.right = None
        else:
            self.best_ind, self.split_value, true_class_features, true_class_labels,false_class_features,false_class_lables  = self.points.split_with_best_gain()
        
            self.right=Tree(false_class_features,false_class_lables,types,h-1)
            self.left=Tree(true_class_features,true_class_labels,types,h-1)
            self.node = best_feat
    
    
    def decide(self, features: List[float]):
        if self.left is None and self.right is None:
            # When we reach a leaf node, we determine the majority class
            negative = 0
            positive = 0
            for i in self.points.labels:
                if i:
                    positive += 1
                else:
                    negative += 1
            if negative >= positive:
                return False  # Class 0 is the majority
            else:
                return True  # Class 1 is the majority

        if(self.points.types[self.best_ind]== FeaturesTypes.BOOLEAN):
            if features[self.node]:
                return self.left.decide(features)  # Recurse to the left child
            else:
                return self.right.decide(features)  # Recurse to the right child
            
        elif(self.points.types[self.best_ind]== FeaturesTypes.CLASSES):
            if features[self.node] == self.split_value:
                return self.left.decide(features)  # Recurse to the left child
            else:
                return self.right.decide(features)  # Recurse to the right child
    

        