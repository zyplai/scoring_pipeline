from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF
from pandas.api.types import is_numeric_dtype, is_string_dtype
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from configs.config import settings
from validation.psi import get_all_psi
from validation.adversarial_val import perform_adv_val
from validation.ks_test import compare_datasets
from wonderwords import RandomWord
import shap


def create_report_fpdf(data, model, runtime) -> None:
    alias = get_random_alias()
    
    threshold_breakdown, data_breakdown, model_performance = get_performance_metric(
        data
    )

    doc_font = 'Arial'
    section_split_space = 10
    runtime = datetime.strptime(runtime, '%Y-%m-%d_%H-%M-%S')

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=1.0) # max width is 190

    pdf.set_fill_color(r=240, g=240, b=240)
    pdf.set_draw_color(r=200, g=200, b=200)

    #################################################################################
    # Metadata:

    pdf.set_font(doc_font, style='B', size=16)
    pdf.cell(w=0, h=10, txt='Training run: ' + alias, border='B', ln=1)
    pdf.ln(h=2)

    pdf.set_font(doc_font, style='B', size=10)
    pdf.cell(w=pdf.get_string_width('Date: '), h=5, txt='Date: ', ln=0)
    pdf.set_font(doc_font, style='', size=10)
    pdf.cell(
        w=pdf.get_string_width(str(runtime)), h=5, txt=str(runtime), ln=1
    )

    pdf.set_font(doc_font, style='B', size=10)
    pdf.cell(w=pdf.get_string_width('Author: '), h=5, txt='Author: ', ln=0)
    pdf.set_font(doc_font, style='', size=10)
    pdf.cell(
        w=pdf.get_string_width(settings.METADATA.author_name), h=5, txt=settings.METADATA.author_name, ln=1
    )

    pdf.set_font(doc_font, style='B', size=10)
    pdf.cell(w=pdf.get_string_width('Partner: '), h=5, txt='Partner: ', ln=0)
    pdf.set_font(doc_font, style='', size=10)
    pdf.cell(
        w=pdf.get_string_width(settings.FEATURES_PARAMS.partner_name),
        h=5,
        txt=settings.FEATURES_PARAMS.partner_name,
        ln=1,
    )
    
    pdf.set_font(doc_font, style='B', size=10)
    pdf.cell(w=pdf.get_string_width('Target mean encoding: '), h=5, txt='Target mean encoding: ', ln=0)
    pdf.set_font(doc_font, style='', size=10)
    pdf.cell(
        w=10,
        h=5,
        txt="Enabled" if settings.TARGET_MEAN_ENCODE.target_encode else "Disabled",
        ln=1,
    )
    
    pdf.set_font(doc_font, style='B', size=10)
    pdf.cell(w=pdf.get_string_width('Macro enrichment: '), h=5, txt='Macro enrichment: ', ln=0)
    pdf.set_font(doc_font, style='', size=10)
    pdf.cell(
        w=10,
        h=5,
        txt="Enabled" if settings.MACRO.enrichment else "Disabled",
        ln=1,
    )

    pdf.ln(h=section_split_space - 7)

    #################################################################################
    # Data overview:
    feature_info = get_feature_report(data=data)

    pdf.set_font(doc_font, 'B', 13)
    pdf.cell(w=0, h=10, txt='Data overview', border='T', ln=1)
    
    ##### Train/test split ############################
    pdf.set_font(family=doc_font, style='B', size=8)
    # Iterate over COLUMNS and display them
    for key, value in data_breakdown.items():
        pdf.cell(w=190/11, h=7, txt=key, border=1, ln=0, fill=True)

    pdf.ln()

    # Iterate over VALUES and display them
    pdf.set_font(family=doc_font, style='', size=8)
    for key, value in data_breakdown.items():
        pdf.cell(w=190/11, h=7, txt=str(value), border=1, ln=0)
        
    pdf.ln(section_split_space)
    #################################################
    
    ####### Feature info ############################
    pdf.set_font(doc_font, style='B', size=8)
    # Iterate over COLUMNS and display them
    for col in feature_info.columns:
        pdf.cell(w=190/6, h=7, txt=str(col), border=1, ln=0, fill=True)

    pdf.ln()

    # Iterate over VALUES and display them
    pdf.set_font(family=doc_font, style='', size=8)
    for _, row in feature_info.iterrows():
        for col in feature_info.columns:
            pdf.cell(w=190/6, h=7, txt=str(row[col]), border=1, ln=0, fill=False)
        pdf.ln()

    #################################################################################
    # Model performance and hyperparameters:

    pdf.ln(h=section_split_space - 7)
    pdf.set_font(doc_font, 'B', 13)
    pdf.cell(w=0, h=10, txt='Model performance and hyperparameters', border='T', ln=1)

    pdf.set_font(family=doc_font, style='B', size=8)

    # Iterate over COLUMNS and display them
    for key, value in model_performance.items():
        pdf.cell(w=22, h=7, txt=key, border=1, ln=0, fill=True)

    pdf.ln()

    # Iterate over VALUES and display them
    pdf.set_font(family=doc_font, style='', size=8)
    for key, value in model_performance.items():
        pdf.cell(w=22, h=7, txt=str(value), border=1, ln=0)
        
    ########### Model hyperparameters #########

    pdf.ln(h=section_split_space)
    pdf.set_font(family=doc_font, style='B', size=8)

    # Iterate over COLUMNS and display them
    for key, value in settings.SET_FEATURES.model_params.items():
        pdf.cell(w=22, h=7, txt=key, border=1, ln=0, fill=True)

    pdf.ln()

    # Iterate over VALUES and display them
    pdf.set_font(family=doc_font, style='', size=8)
    for key, value in settings.SET_FEATURES.model_params.items():
        pdf.cell(w=22, h=7, txt=str(value), border=1, ln=0)


    #################################################################################
    # Threshold breakdown:

    pdf.ln(h=section_split_space)
    pdf.set_font(doc_font, 'B', 13)
    pdf.cell(w=0, h=10, txt='Threshold breakdown', border='T', ln=1)

    pdf.set_font(family=doc_font, style='B', size=8)

    # Iterate over COLUMNS and display them
    for col in threshold_breakdown.columns:
        if str(col) in ['TN', 'TP', 'FN', 'FP']:
            pdf.cell(w=14, h=7, txt=str(col), border=1, ln=0, fill=True)
        else:
            pdf.cell(w=(190-14*4)/6, h=7, txt=str(col), border=1, ln=0, fill=True)

    pdf.ln()

    # Iterate over VALUES and display them
    pdf.set_font(family=doc_font, style='', size=8)
    for _, row in threshold_breakdown.iterrows():
        for col in threshold_breakdown.columns:
            if str(col) in ['TN', 'TP', 'FN', 'FP']:
                pdf.cell(w=14, h=7, txt=str(int(row[col])), border=1, ln=0, fill=False)
            else:
                pdf.cell(w=(190-14*4)/6, h=7, txt=str(row[col]), border=1, ln=0, fill=False)
        pdf.ln()
        
    
    ###### Optimal threshold ######
    pdf.ln(section_split_space-7)
    
    opt_thres = get_optimal_threshold(data)
    
    pdf.set_font(doc_font, style='B', size=10)
    pdf.cell(w=120, 
             h=5, 
             txt='Optimal threshold (Youden\'s J statistic): ', align='R', ln=0)
    pdf.set_font(doc_font, style='', size=10)
    pdf.cell(
        w=pdf.get_string_width(str(opt_thres)), h=5, txt=str(opt_thres), ln=1
    )

    #################################################################################
    # Kolmogorov Smirnov test:

    ks_result = compare_datasets(
        data[data['is_train'] == 1], data[data['is_train'] == 0]
    )

    # Convert to DataFrame
    ks_result_df = pd.DataFrame(
        [(col, data['p_value'], data['similarity']) for col, data in ks_result.items()],
        columns=['Feature', 'P-value', 'Similarity'],
    ).sort_values(by='P-value', ascending=True)

    ks_result_df['Similarity'] = ks_result_df['Similarity'].str.replace(
        'Likely from the same distribution', 'High'
    )
    ks_result_df['Similarity'] = ks_result_df['Similarity'].str.replace(
        'Likely from different distribution', 'Low'
    )

    pdf.ln(h=section_split_space - 7)
    pdf.set_font(doc_font, 'B', 13)
    pdf.cell(w=0, h=10, txt='Train/test split validations', border='T', ln=1)

    pdf.set_font(family=doc_font, style='B', size=10)
    pdf.cell(w=190, h=7, txt=str('Kolmogorov Smirnov test'), border=1, ln=1, fill=True)

    pdf.set_font(family=doc_font, style='B', size=8)
    # Iterate over COLUMNS and display them
    for col in ks_result_df.columns:
        pdf.cell(w=190/3, h=7, txt=str(col), border=1, ln=0, fill=True)

    pdf.ln()

    # Iterate over VALUES and display them
    pdf.set_font(family=doc_font, style='', size=8)
    for _, row in ks_result_df.iterrows():
        for col in ks_result_df.columns:
            if col == 'P-value':
                pdf.cell(
                    w=190/3, h=7, txt=str(round(row[col], 4)), border=1, ln=0, fill=False
                )
            else:
                pdf.cell(w=190/3, h=7, txt=str(row[col]), border=1, ln=0, fill=False)
        pdf.ln()

    #################################################################################
    # Adversarial validation:

    adv_val_result = perform_adv_val(data)

    pdf.ln(h=section_split_space - 7)
    pdf.set_font(family=doc_font, style='B', size=10)
    pdf.cell(w=90, h=7, txt=str('Adversarial validation'), border=1, ln=1, fill=True)
    pdf.cell(w=45, h=7, txt=str('AUC'), border=1, ln=0)
    pdf.cell(w=45, h=7, txt=str(adv_val_result), border=1, ln=0)
    
    #################################################################################
    # PSI:

    psi_result = get_all_psi(
        data[data['is_train'] == 1], 
        data[data['is_train'] == 0], 
        list(settings.SET_FEATURES.cat_feature_list), 
        '')
    
    pdf.ln(h=section_split_space)
    pdf.set_font(doc_font, 'B', 13)
    pdf.cell(w=0, h=10, txt='PSI', border='T', ln=1)

    for key, feature in psi_result.items():
        pdf.set_font(doc_font, style='B', size=8)

        # Iterate over COLUMNS and display them
        for col in feature.columns:
            pdf.cell(w=190/4, h=7, txt=str(col)[:32], border=1, ln=0, fill=True)

        pdf.ln()

        # Iterate over VALUES and display them
        pdf.set_font(family=doc_font, style='', size=8)
        for _, row in feature.iterrows():
            for col in feature.columns:
                pdf.cell(w=190/4, h=7, txt=str(row[col])[:32], border=1, ln=0, fill=False)
            pdf.ln()
            
        pdf.ln(h=2)
    
    #################################################################################
    # Model feature importance:

    png_path = get_feature_importance(data, model)

    pdf.ln(h=section_split_space)
    pdf.set_font(doc_font, 'B', 13)
    pdf.cell(w=0, h=10, txt='Model feature importance', border='T', ln=1)

    pdf.image(png_path, w=190, type='png', link='')
    
    #################################################################################
    # Shapley feature importance:

    png_path_shap = get_shapley_feature_importance(data, model)
    
    pdf.add_page()
    pdf.set_font(doc_font, 'B', 13)
    pdf.cell(w=0, h=10, txt='Shapley feature importance', border='T', ln=1)

    pdf.image(png_path_shap, w=190, type='png', link='')

    pdf.output('data/model_report.pdf', 'F')


def get_feature_importance(data, clf):
    path = 'data/catboost_feature_importance.png'
    if settings.TARGET_MEAN_ENCODE.target_encode:
        features = data[settings.SET_FEATURES.features_list_tme].columns
    else:
        features = data[settings.SET_FEATURES.features_list].columns
        
    feature_importance = clf.feature_importances_
    sorted_idx = np.argsort(feature_importance)

    plt.figure(figsize=(12, 6))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(features)[sorted_idx])
    plt.savefig(path, bbox_inches='tight')
    plt.close()

    return path

def get_shapley_feature_importance(data, clf):
    path = 'data/shapley_feature_importance.png'
    
    if settings.TARGET_MEAN_ENCODE.target_encode:    
        X = data[settings.SET_FEATURES.features_list_tme]
    else:
        X = data[settings.SET_FEATURES.features_list]
    
    y = data['target']
    
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X, y)
    
    shap.plots.violin(shap_values, features=X, feature_names=list(X.columns), plot_type="layered_violin", show=False)
    plt.savefig(path, bbox_inches='tight', dpi=400)
    
    return path


def get_random_alias():
    r = RandomWord()
    adj = r.word(include_parts_of_speech=["adjectives"], word_min_length=3, word_max_length=8)
    noun = r.word(include_parts_of_speech=["nouns"], word_min_length=3, word_max_length=8)
    return str(adj+"-"+noun)


def get_optimal_threshold(data):
    y_test = data[data['is_train'] == 0]['target']
    preds = data[data['is_train'] == 0]['predictions']
    
    fpr, tpr, thresholds = roc_curve(y_test, preds)

    # Find the optimal threshold that maximizes the Youden's J statistic
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = round(thresholds[optimal_idx], 3)
    return optimal_threshold


def get_performance_metric(preds: pd.DataFrame):
    """
    Calculate model performance metrics

    Args:
        preds (pd.DataFrame): Model predictions

    Returns:
        pd.DataFrames
    """
    train_set = preds[preds['is_train'] == 1].copy()
    test_set = preds[preds['is_train'] == 0].copy()
    target = 'target'

    model_threshold_list = []

    general_info = {
        'Samples': train_set.shape[0] + test_set.shape[0],
        'Train size': train_set.shape[0],
        'Test size': test_set.shape[0],
        'Total good': train_set[target].value_counts()[0]
        + test_set[target].value_counts()[0],
        'Total bad': train_set[target].value_counts()[1]
        + test_set[target].value_counts()[1],
        'Train good': train_set[target].value_counts()[0],
        'Train bad': train_set[target].value_counts()[1],
        'Test good': test_set[target].value_counts()[0],
        'Test bad': test_set[target].value_counts()[1],
        'Train NPL': round(train_set[target].value_counts()[1] / train_set.shape[0], 4),
        'Test NPL': round(test_set[target].value_counts()[1] / test_set.shape[0], 4),
    }

    model_performance = {
        'Train AUC': round(
            roc_auc_score(train_set[target], train_set['predictions']), 3
        ),
        'Test AUC': round(roc_auc_score(test_set[target], test_set['predictions']), 3),
        'Train Gini': round(
            2 * round(roc_auc_score(train_set[target], train_set['predictions']), 3)
            - 1,
            3,
        ),
        'Test Gini': round(
            2 * round(roc_auc_score(test_set[target], test_set['predictions']), 3) - 1,
            3,
        ),
    }

    for thres in [
        x / 100
        for x in range(
            settings.THRESHOLD_REPORT.min_thres,
            settings.THRESHOLD_REPORT.max_thres + 1,
            settings.THRESHOLD_REPORT.stp_thres,
        )
    ]:
        thres_metrics = dict()

        test_set['prediction_label'] = (test_set['predictions'] > thres).astype(int)

        thres_metrics['Threshold'] = thres
        thres_metrics['Accuracy'] = round(
            accuracy_score(test_set[target], test_set['prediction_label']), 3
        )

        tn, fp, fn, tp = confusion_matrix(
            test_set[target], test_set['prediction_label']
        ).ravel()
        thres_metrics['Model NPL'] = round(fn / (tn + fn), 3)
        thres_metrics['Approval rate'] = round((fn + tn) / (tn + fp + fn + tp), 3)
        thres_metrics['Good found'] = round(tn / (fp + tn), 3)
        thres_metrics['Bad found'] = round(tp / (fn + tp), 3)

        thres_metrics['TP'] = tp
        thres_metrics['TN'] = tn
        thres_metrics['FP'] = fp
        thres_metrics['FN'] = fn

        model_threshold_list.append(thres_metrics)

    return pd.DataFrame(model_threshold_list), general_info, model_performance


def get_feature_report(data: pd.DataFrame):
    feature_report = list()
    for column in settings.SET_FEATURES.features_list:
        if column in list(data.columns):
            column_info = dict()
            column_info['Feature'] = str(column)[:22]
            column_info['Type'] = data[column].dtype
            column_info['Missing'] = round(data[column].isnull().sum() / len(data), 5)

            if is_numeric_dtype(data[column]):
                column_info['Min'] = round(data[column].min(), 5)
                column_info['Max'] = round(data[column].max(), 5)
                column_info['Mode'] = round(data[column].mode()[0], 5)
            elif is_string_dtype(data[column]):
                column_info['Min'] = '-'
                column_info['Max'] = '-'
                column_info['Mode'] = str(data[column].mode()[0])[:22]

            feature_report.append(column_info)

    return pd.DataFrame(feature_report)
