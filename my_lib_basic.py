import os
import subprocess
from pathlib import Path
import SimpleITK as sitk
import radiomics.featureextractor as FEE
import logging
import sys
import numpy as np
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score, roc_curve, confusion_matrix
import pandas as pd
import scipy.stats

PATH_TEMP = Path('G:/5-temp')


def make_path(path):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

def path_replace(old_path, old, new):
    new_path = str(old_path).replace(old, new)
    new_path = Path(new_path)
    return new_path

def make_registeration(path_dst_series, path_old, path_new, show_on=False):
    '''

    Parameters
    ----------
    path_dst_series: Path object
        Full path of series which to be aligned with
    path_old: Path object
        Full path of series which to be aligned
    path_new: Path object
        Full path of series to be written
    show_on: Bool

    Returns
    -------
        None
    '''
    if show_on:
        cmd = [r'G:\3-project\my_libs\resample_toRefrenceImg\Resampler.exe',
               str(path_dst_series), str(path_old), str(path_new), "1"]
    else:
        cmd = [r'G:\3-project\my_libs\resample_toRefrenceImg\Resampler.exe',
               str(path_dst_series), str(path_old), str(path_new)]
    subprocess.call(cmd)

def get_params_path():
    return Path('G:/3-project/my_libs/Params.yaml')

def get_extractor(is_addProvenance=False):
    extractor = FEE.RadiomicsFeaturesExtractor(str(get_params_path()))
    extractor.addProvenance(is_addProvenance)
    return extractor


class MedImg(object):
    def __init__(self, path):
        self.path = path
        self.img = sitk.ReadImage(str(self.path))

    def get_img(self):
        return self.img

    def get_data(self):
        return sitk.GetArrayFromImage(self.img)

    def get_size(self):
        return self.img.GetSize()

    def get_depth(self):
        return self.img.GetDepth()

    def get_path(self):
        return self.path

    def get_path_string(self):
        return str(self.path)



import logging
import sys
class Log(logging.Logger):
    '''
    An easy implement of logging module

    Logging levels:
        DEBUG = 10
        INFO = 20
        WARNNING = WARN = 30
        ERROR = 40

    Logger with level >= 10 is saved in files
    Logger with level >= 20 is output in screen
    '''
    def __init__(self, name='basic', file_path='./work_log'):
        super(Log, self).__init__(name)

        self.setLevel(level=logging.DEBUG)

        # handler 1
        handler1 = logging.FileHandler(str(file_path))
        handler1.setLevel(level=logging.DEBUG)
        formatter1 = logging.Formatter('%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s\n%(message)s')
        handler1.setFormatter(formatter1)
        self.addHandler(handler1)
        # handler 2
        handler2 = logging.StreamHandler(sys.stdout)
        handler2.setLevel(level=logging.INFO)
        formatter2 = logging.Formatter('%(asctime)s - %(lineno)s - %(levelname)s\n%(message)s')
        handler2.setFormatter(formatter2)
        self.addHandler(handler2)

    def start(self):
        self.debug('\n************************** Log Start **************************')

    def end(self):
        self.debug('\n************************** Log End **************************\n\n\n\n')

def evaluate(y_true, y_pred, cutoff=0.5):
    '''
    calculate several metrics of predictions

    Args:
        y_true: list, labels
        y_pred: list, predictions
        cutoff: float

    Returns:
        evaluation: dict

    '''
    y_pred_t = [1 if i > cutoff else 0 for i in y_pred]

    evaluation = OrderedDict()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_t).ravel()
    evaluation['auc'] = roc_auc_score(y_true, y_pred)
    evaluation['acc'] = accuracy_score(y_true, y_pred_t)
    evaluation['recall'] = recall_score(y_true, y_pred_t)
    evaluation['f1'] = f1_score(y_true, y_pred_t)
    evaluation['specificity'] = tn / (tn + fp)
    evaluation['cutoff'] = cutoff

    return evaluation

def draw_roc(y_true, y_pred):
    fpr_test, tpr_test, _ = roc_curve(label_test, pred_test)
    auc = roc_auc_score(label_test, pred_test)
    plt.plot(fpr_test, tpr_test, label=f'{model} ROC curve (area = {auc:0.3f})')


def find_best_cutoff(y_true, y_pred):
    '''
    search the optimal cutoff based on the highest SEN+SPE in training cohort

    Args:
        y_true: list, labels
        y_pred: list, predictions

    Returns:
        cutoff_best: float

    '''
    fpr, tpr, cutoffs = roc_curve(y_true, y_pred, drop_intermediate=False)
    index = np.argmax(tpr - fpr)
    cutoff_best = cutoffs[index]

    return cutoff_best
