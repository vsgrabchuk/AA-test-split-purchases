import pandas as pd
import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def synth_test(
    s1,
    s2,
    ss_percent=10,
    n_simulations=1000,
    synth_p_val_th=0.05,
    fpr_norm_err=0.01,
    test='t'
):
    '''
    Функция, производящая синтетический тест на 2х выборках
    Синтетический тест - сравнение множества подвыборок без повторений
    
    Parameters
    ----------
    s1: pandas.Series
        Выборка 1 (sample)
    s2: pandas.Series
        Выборка 2
    ss_percent: float, default 10
        Процент от выборки (min размера) для составления подвыборки (subsample)
    n_simulations: int, default 1000
        Количество симуляций
    synth_p_val_th: float, default 0.05
        Порог значимости для синтетического p-value (для расчёта FPR)
    fpr_norm_err: float, default 0.01
        Допустимая ошибка для FPR (FPR +- fpr_norm_err -> ok)
    test: str, default 't'
        Статистический тест
        't' - t-тест
        'u' - тест Манна-Уитни
    
    Returns
    -------
    fpr_ok: bool
        Впорядке ли значение FPR, предполагая, что выборки одинаковые
    '''
    n_s_min = min(len(s1), len(s2))  # Минимальный размер из выборок
    n_ss = round(n_s_min * ss_percent / 100)  # Количество элементов в подвыборке
    
    print('min sample size:', n_s_min)
    print('synthetic subsample size:', n_ss)
    
    p_vals = []  # Список с p-values
    
    # Цикл симуляций с отображением статусбара
    for i in tqdm(range(n_simulations)):
        ss1 = s1.sample(n_ss, replace = False)
        ss2 = s2.sample(n_ss, replace = False)
        
        if test == 't':  # t-тест с поправкой Уэлча
            test_res = st.ttest_ind(ss1, ss2, equal_var=False)
        elif test == 'u':  # U-тест
            test_res = st.mannwhitneyu(ss1, ss2)
            
        p_vals.append(test_res[1]) # Сохраняем p-value
        
    # Визулаилзация распределения p-values
    plt.hist(p_vals, bins = 50)
    plt.style.use('ggplot')
    plt.xlabel('p-values')
    plt.ylabel('frequency')
    plt.title("Histogram of synthetic simulations")
    plt.show()
    
    # Доля исходов со статзначимыми различиями
    # В контексте A/A-test - FPR
    fpr = sum(np.array(p_vals) <= synth_p_val_th) / n_simulations
    
    if fpr < (synth_p_val_th + fpr_norm_err):
        fpr_ok = True
    else:
        fpr_ok = False
        
    print('FPR:', fpr)
    print('fpr_ok:', fpr_ok)
    
    return fpr_ok