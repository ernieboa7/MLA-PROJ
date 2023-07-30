import numpy as np

def subset_quality(feature_set):
    feature_set = np.array(feature_set)
    u = np.sum(np.sin(feature_set)+ np.cos(feature_set))
    return u


Selected = []
Features = [1, 2, 3, 4, 5, 6]

while len(Selected) < 3:
    print('Selected ', Selected)
    print('Features ', Features)

    Best = -1
    FBest = -1
    for i in Features:
        feature_set = list([i])
        feature_set.extend(Selected)
        ISet = [0, 0, 0, 0, 0, 0]
        for j in feature_set:
            ISet[j-1] = 1
            
        Fn = subset_quality(feature_set)
        print('  Testing ', feature_set, ', F Value ', Fn)
        if Fn > FBest:
            Best = i
            FBest = Fn
    
    if Best != -1:
        Selected.append(Best)
        Features.remove(Best)
        
print('\nFinal Selection ', Selected)
print('Fn Value ', FBest)


