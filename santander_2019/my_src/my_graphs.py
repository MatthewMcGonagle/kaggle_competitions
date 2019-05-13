import matplotlib.pyplot as plt
from matplotlib.lines import Line2D # Needed to create custom legends using dummy lines.
import seaborn as sns
from sklearn.metrics import roc_curve

def graph_in_class_distribution(feature, X, y):
    for target in [0, 1]:
        on_target = y.loc[X.index] == target
        sns.distplot(X.loc[on_target, feature])
    plt.legend(['class ' + str(i) for i in range(2)])
    plt.xlabel('Reduced Feature ' + str(feature))
    plt.ylabel('In-class Proportion')
    
# For large data, it is convenient to graph a sample drawn from the data.

def graph_2d_in_class_distribution(features, X, y, sample_size = 10**4):

    for target in [0, 1]:
        on_target = y.loc[X.index] == target
        if sample_size is None:
            data = X.loc[on_target, features]
        else:
            data = X.loc[on_target, features].sample(sample_size)
        sns.kdeplot(data[features[0]], data[features[1]], shade_lowest = False)

    # Create dummy lines to make a custom legend.
    class_lines = [Line2D([0], [0], color = sns.color_palette()[0]), #'blue'),
               Line2D([0], [0], color = sns.color_palette()[1])]
    plt.legend(class_lines, ['class ' + str(i) for i in range(2)])
    plt.xlabel('Reduced Feature ' + str(features[0]))
    plt.ylabel('Reduced Feature ' + str(features[1]))

def find_cv_roc_curves(predictor, kfold_X, y):
    roc_curves = []
    for X in kfold_X:
        predictor.fit(X['train'], y[X['train'].index])
        y_predict = predictor.predict_proba(X['test'])[:, 1]
        y_true = y[X['test'].index]
        curve = roc_curve(y_true, y_predict)
        roc_curves.append(curve)   
    return roc_curves

def graph_roc_curves(roc_curves, stride = 10):
    for curve in roc_curves:
        false_pos, true_pos, _ = curve
        plt.plot(false_pos[::stride], true_pos[::stride]) # Sub-sample to save memory for file.
    plt.legend(['fold ' + str(i) for i, _ in enumerate(roc_curves)])
    plt.title('Cross-fold Validated ROC Curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
