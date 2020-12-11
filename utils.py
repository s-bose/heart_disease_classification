import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
import itertools
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, f1_score


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    IMAGES_PATH = os.path.join(os.getcwd(), 'images')
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

    
    
def plot_all_pair(X, y, columns, target, lower, colormap, title):
    sns.set_palette(sns.color_palette(colormap))
    df = pd.DataFrame(X, columns=columns)
    df[target] = y
    graph = sns.pairplot(df, x_vars=columns, y_vars=columns, hue=target)
    graph.fig.suptitle(title)
    
    def hide_current_axis(*args, **kwds):
        plt.gca().set_visible(False)
    if (lower is True):
        graph.map_upper(hide_current_axis)
        

def plot_pair(X, y, columns, target, colormap, title):
    sns.set_palette(sns.color_palette(colormap))
    df = pd.DataFrame(X, columns=columns)
    df[target] = y
    graph = sns.scatterplot(data=df, x=columns[0], y=columns[1], hue=target)
    plt.title(title)
    plt.show()
    

def plot_accuracy(model, X_train, X_test, y_train, y_test, model_name):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    f1_train = f1_score(y_train, y_train_pred)
    f1_test = f1_score(y_test, y_test_pred)
    
    print(f'Train accuracy for {model_name}: {model.score(X_train, y_train)}, F1 score: {f1_train}')
    print(f'Test accuracy for {model_name}: {model.score(X_test, y_test)}, F1 score: {f1_test}')
    f, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].set_title("Train Set")
    ax[1].set_title("Test Set")
    plot_confusion_matrix(model, X_train, y_train, cmap="Blues", ax=ax[0])
    plot_confusion_matrix(model, X_test, y_test, cmap="Blues", ax=ax[1])

def plot_accuracy_sns(model, X_train, X_test, y_train, y_test, model_name):
    '''
    An alternate implementation of plotting confusion matrix and 
    printing Training and Testing accuracy.
    This one does not use plot_confusion_matrix from sklearn but uses
    seaborn heatmap plot.
    '''
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    f1_train = f1_score(y_train, y_train_pred)
    f1_test = f1_score(y_test, y_test_pred)
    
    print(f'Train accuracy for {model_name}: {model.score(X_train, y_train)}, F1 score: {f1_train}')
    print(f'Test accuracy for {model_name}: {model.score(X_test, y_test)}, F1 score: {f1_test}')
    
    fig, ax =plt.subplots(1,2, figsize=(10,3))
    ax[0].set_title("Train Set Accuracy")
    ax[1].set_title("Test Set Accuracy")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    confusion_train = confusion_matrix(y_train, y_train_pred)
    confusion_test = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(confusion_train, annot=True, fmt='g', ax=ax[0])
    sns.heatmap(confusion_test, annot=True, fmt='g', ax=ax[1])
    
    ax[0].set_xlabel("Predicted Label")
    ax[1].set_xlabel("Predicted Label")
    ax[0].set_ylabel("True Label")
    ax[1].set_ylabel("True Label")
    
    