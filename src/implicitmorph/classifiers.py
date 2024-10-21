import logging
import numpy as np
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import matplotlib.pylab as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib').disabled = True
logging.getLogger().setLevel(logging.INFO)

ri_label_names = {0: 'neuron', 1: 'reconstruction_issue'}
ritype2int = {
    'exc': 0,
    'inh': 0,
    'ri': 1,
}

ei_label_names = {0: 'exc', 1: 'inh'}
eitype2int = {
    'exc': 0,
    'inh': 1,
}

layer_label_names = {0: 'L23', 1: 'L4', 2: 'L5', 3: 'L6'}
layer2int = {'L23': 0, 'L4': 1, 'L5': 2, 'L6': 3}

celltype_label_names = {0: '23P', 1: '4P', 2: '5P-IT', 3: '5P-NP', 4: '5P-PT', 5: '6P-CT', 6: '6P-IT'}#, 7: 'WM-P'}
celltype2int = {'23P': 0, '4P': 1, '5P-IT': 2, '5P-NP': 3, '5P-PT': 4, '6P-CT': 5, '6P-IT': 6}#, 'WM-P': 7}


def run_ri_classifier(df_classifier, df, path, embedding_name='latent_emb'):
    '''Classifier: Whole versus incomplete neuron.
    '''
    # Prepare data.
    logging.info(f'Embedding type: {embedding_name}')
    latents = np.stack(df_classifier[embedding_name].values).astype(float)
    labels =  np.stack([ritype2int[label] for label in df_classifier['cell_type_coarse'].values])

    df = run_classifier_pipeline(latents, labels, 'accuracy', df, ri_label_names, 'ri', path)

    return df


def run_ei_classifier(df_classifier, df, path, embedding_name='latent_emb'):
    '''Classifier: Excitatory versus inhibitory neuron.
    '''
    df_classifier = df_classifier[df_classifier['cell_type_coarse'] != 'ri']
    
    # Prepare data.
    logging.info(f'Embedding type: {embedding_name}')
    latents1 = np.stack(df_classifier[embedding_name].values).astype(float)
    latents2 = np.stack(df_classifier[['syn_density_shaft_after_proof', 'spine_density_after_proof']].values).astype(float)
    latents = np.concatenate([latents1, latents2], axis=1)
    
    labels = np.stack([eitype2int[label] for label in df_classifier['cell_type_coarse'].values])

    df = run_classifier_pipeline(latents, labels, 'balanced_accuracy', df, ei_label_names, 'cell_type_coarse', path, embedding_name)
    
    return df


def run_layer_classifier(df_classifier, df, path, embedding_name='latent_emb'):
    '''Classifier: Cortical layers.
    '''
    df_classifier = df_classifier[(df_classifier['cell_type_coarse'] == 'exc')]
    
    # Prepare data.
    logging.info(f'Embedding type: {embedding_name}')
    latents = np.stack(df_classifier[embedding_name].values).astype(float)
    labels = np.stack([layer2int[label] for label in df_classifier['layer'].values])

    df = run_classifier_pipeline(latents, labels, 'balanced_accuracy', df, layer_label_names, 'layer', path, embedding_name)
    
    return df


def run_cell_type_classifier(df_classifier, df, path, embedding_name='latent_emb'):
    '''Classifier: Excitatory cell type.
    '''
    df_classifier = df_classifier[(df_classifier['cell_type_coarse'] == 'exc')]
    
    # Prepare data.
    logging.info(f'Embedding type: {embedding_name}')
    latents = np.stack(df_classifier[embedding_name].values).astype(float)
    labels = np.stack([celltype2int[label] for label in df_classifier['cell_type'].values])

    df = run_classifier_pipeline(latents, labels, 'balanced_accuracy', df, celltype_label_names, 'cell_type', path, embedding_name)
    
    return df


def run_classifier_pipeline(latents, labels, metric, df, label_names, classification_type, path, embedding_name):
    out_path = Path(path, 'classifier')
    out_path.mkdir(parents=True, exist_ok=True)

    logging.info(f'Run cross-validation for {classification_type}-classifier.')
    scores, mean_score, std_score, predictions, gt_labels = run_crossvalidation(latents, labels, metric)

    logging.info(f'mean score: {mean_score:.4f}')
    for s in scores:
        logging.info(f'score: {s:.3f} ')

    logging.info('Plot confusion matrix.')
    df_cm = pd.DataFrame()
    df_cm[f'{classification_type}_cm_prediction'] = predictions
    df_cm[f'{classification_type}_cm_labels'] = gt_labels
    df_cm.to_pickle(Path(out_path, f'{classification_type}_cm.pkl'))
    plot_confusion(predictions, gt_labels, label_names, classification_type, out_path)
        
    file = Path(out_path, f'classifier_{classification_type}.txt')
    with open(file, 'w') as f:
        f.write(f'score {mean_score}\n')
        f.write(f'std {std_score}\n')
        for i, s in enumerate(scores):
            f.write(f'score {i}: {s}\n')
    
    return df


def run_crossvalidation(latents, labels, metric):
    
    assert metric in ['accuracy', 'balanced_accuracy']

    SPLITS = 10
    RANDOM = 42

    # Set up possible values of parameters to optimize over
    svm_grid = {"C": [0.5, 1, 3, 5, 10, 20, 30], "gamma": [0.01, 0.1], "degree": [2, 3, 5, 7, 10], "class_weight": [None, 'balanced'], "kernel": ['rbf', 'poly']}
    svm = SVC(random_state=RANDOM)

    rf_grid = {"max_depth": [2, 3, 5, 7, 10, 15, 20]}
    rf = RandomForestClassifier(random_state=RANDOM)

    lr_grid = {"C": [0.5, 1., 3., 5., 10., 20., 30.], "penalty": ['none', 'l2', 'l1', 'elasticnet'], "class_weight": [None, 'balanced']}
    lr = LogisticRegression(solver='saga', l1_ratio=0.5, random_state=RANDOM)

    estimators = [svm, rf, lr]
    grids = [svm_grid, rf_grid, lr_grid]

    outer_cv = StratifiedKFold(n_splits=SPLITS, shuffle=True, random_state=RANDOM)
    outer_scores = np.zeros(SPLITS)
    preds, gt = [], []

    for i, (train_outer_index, test_outer_index) in enumerate(tqdm(outer_cv.split(latents, labels), total=outer_cv.get_n_splits(), desc="k-fold")):
        scores = np.zeros(len(estimators))
        predictors = []
        inner_cv = StratifiedKFold(n_splits=SPLITS, shuffle=True, random_state=RANDOM)
        for k, (est, grid) in enumerate(tqdm(zip(estimators, grids), total=len(estimators), desc="optim")):
            clf = GridSearchCV(estimator=est, param_grid=grid, cv=inner_cv).fit(latents[train_outer_index], labels[train_outer_index])
            scores[k] = clf.best_score_
            predictors.append(clf)

        idx = np.argmax(scores)
        outer_scores[i] = predictors[idx].score(latents[test_outer_index], labels[test_outer_index])
        preds.append(predictors[idx].predict(latents[test_outer_index]))
        gt.append(labels[test_outer_index])
    preds, gt = np.concatenate(preds), np.concatenate(gt)

    return outer_scores, outer_scores.mean(), outer_scores.std(), preds, gt


def plot_confusion(predictions, labels, label_names, classification_type, out_path=None):
    ''' Plot confusion matrix.
    '''
    cm = confusion_matrix(labels, predictions)
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    fig, ax = plt.subplots(1, 1)
    fig.set_figwidth(len(cm[0]) * 1)
    fig.set_figheight(len(cm) * 1)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.OrRd, ax=ax) 
    ax.set_xticklabels(list(label_names.values()), rotation='vertical')
    ax.set_yticklabels(list(label_names.values()))
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.xlabel('Predicted label', fontsize=15)
    plt.ylabel('True label', fontsize=15)
    
    if out_path is not None:
        savepath = Path(out_path, f'{classification_type}_confusion_matrix_norm.pdf')
        fig.savefig(savepath, bbox_inches='tight')
    else:
        plt.show()