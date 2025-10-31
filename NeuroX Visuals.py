# Complete Visualization Graphs Code for NeuroDetect-X Project
# This code generates all the key visualizations mentioned in the paper.
# It assumes you have run the full NeuroDetect-X code and have the necessary variables (e.g., models, predictions, datasets).
# If not, integrate this with the previous code snippets.
# Libraries: matplotlib, seaborn, numpy, pandas, scikit-learn.
# Install if needed: pip install matplotlib seaborn

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Assuming you have these from the full code:
# - X_test, y_test (from primary dataset)
# - y_pred, y_prob (predictions from NeuroDetect-X)
# - Baseline models: svm, rf, xgb, etc. (train and predict similarly)
# - Ablation results: e.g., accuracies, f1_scores, drs for different configs
# - Selected features: selected_features list
# - Training times: e.g., times = {'SVM': 1243, 'RF': 156, 'GBM': 234, 'NeuroDetect-X': 847}
# - Cross-dataset: primary_metrics, validation_metrics
# - Early detection: variants = ['Prodromal PD', 'Mild Tremor', 'Atypical'], accuracies = [91.3, 88.7, 89.4]

# For demo, we'll use placeholders. Replace with actual values.

# Placeholder data (replace with real)
y_test = np.random.choice([0,1,2,3], 570)  # Multi-class
y_pred_neuro = np.random.choice([0,1,2,3], 570)
y_prob_neuro = np.random.rand(570, 4)  # Probabilities for 4 classes

# Baseline predictions (simulate)
y_pred_svm = np.random.choice([0,1,2,3], 570)
y_pred_rf = np.random.choice([0,1,2,3], 570)
y_pred_gbm = np.random.choice([0,1,2,3], 570)
y_pred_xgb = np.random.choice([0,1,2,3], 570)

# Ablation data
configs = ['Without C-GB', 'Without Boruta', 'Without Fusion', 'FB-T Only', 'RFL Only', 'Complete Framework']
accuracies_ab = [0.9123, 0.9278, 0.9345, 0.8967, 0.9034, 0.9662]
f1_scores_ab = [0.9087, 0.9234, 0.9321, 0.8923, 0.8998, 0.9590]
drs_ab = [0.9105, 0.9256, 0.9333, 0.8945, 0.9016, 0.9608]

# Feature distribution
categories = ['Motor', 'Clinical', 'Biomarkers']
counts = [30, 12, 5]  # Example: 47 selected features distributed

# Training times
models_time = ['SVM', 'RF', 'GBM', 'NeuroDetect-X']
times = [1243, 156, 234, 847]

# Cross-dataset
datasets = ['Primary', 'Independent']
acc_cross = [0.9662, 0.9437]
f1_cross = [0.9590, 0.9384]
drs_cross = [0.9608, 0.9412]

# Early detection
variants = ['Prodromal PD', 'Mild Tremor', 'Atypical']
accuracies_ed = [91.3, 88.7, 89.4]

# Figure 1: Methodology Workflow (Text-based diagram, as it's a flowchart)
def plot_workflow():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.9, 'NeuroDetect-X Methodology Workflow', ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(0.1, 0.7, '1. Data Pre-processing\n   - Cluster-Guided Balancing (C-GB)', ha='left', va='center', fontsize=12)
    ax.text(0.1, 0.5, '2. Feature Selection\n   - Boruta Algorithm', ha='left', va='center', fontsize=12)
    ax.text(0.1, 0.3, '3. Classification\n   - Layered Fusion Model\n     * Fast Boost Trees (FB-T)\n     * Randomized Forest Learner (RFL)\n     * Logistic Regression Meta-Learner', ha='left', va='center', fontsize=12)
    ax.arrow(0.3, 0.65, 0, -0.15, head_width=0.02, head_length=0.02, fc='k', ec='k')
    ax.arrow(0.3, 0.45, 0, -0.15, head_width=0.02, head_length=0.02, fc='k', ec='k')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title('Figure 1: NeuroDetect-X Methodology Workflow')
    plt.show()

plot_workflow()

# Figure 2: ROC Curves
def plot_roc_curves():
    fig, ax = plt.subplots(figsize=(8, 6))
    models = ['NeuroDetect-X', 'SVM', 'RF', 'GBM', 'XGBoost']
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, model in enumerate(models):
        if model == 'NeuroDetect-X':
            y_prob = y_prob_neuro[:, 1] if y_prob_neuro.shape[1] > 1 else y_prob_neuro.flatten()
            y_true_bin = label_binarize(y_test, classes=[0,1,2,3])[:, 1]  # Binary for PD vs others
        else:
            y_prob = np.random.rand(570)  # Placeholder
            y_true_bin = label_binarize(y_test, classes=[0,1,2,3])[:, 1]
        fpr, tpr, _ = roc_curve(y_true_bin, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], label=f'{model} (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Figure 2: ROC Curves Comparing NeuroDetect-X and Baseline Models')
    ax.legend()
    plt.show()

plot_roc_curves()

# Figure 3: Precision-Recall Curves
def plot_pr_curves():
    fig, ax = plt.subplots(figsize=(8, 6))
    models = ['NeuroDetect-X', 'SVM', 'RF', 'GBM', 'XGBoost']
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, model in enumerate(models):
        if model == 'NeuroDetect-X':
            y_prob = y_prob_neuro[:, 1]
            y_true_bin = label_binarize(y_test, classes=[0,1,2,3])[:, 1]
        else:
            y_prob = np.random.rand(570)
            y_true_bin = label_binarize(y_test, classes=[0,1,2,3])[:, 1]
        precision, recall, _ = precision_recall_curve(y_true_bin, y_prob)
        ax.plot(recall, precision, color=colors[i], label=model)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Figure 3: Precision-Recall Performance of NeuroDetect-X and Baseline Classifiers')
    ax.legend()
    plt.show()

plot_pr_curves()

# Figure 4: Confusion Matrices
def plot_confusion_matrices():
    models = ['NeuroDetect-X', 'SVM', 'RF', 'GBM', 'XGBoost']
    preds = [y_pred_neuro, y_pred_svm, y_pred_rf, y_pred_gbm, y_pred_xgb]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, (model, pred) in enumerate(zip(models, preds)):
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{model}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    plt.suptitle('Figure 4: Confusion Matrices for NeuroDetect-X and Baseline Classifiers')
    plt.show()

plot_confusion_matrices()

# Figure 5: Model Performance Trends
def plot_performance_trends():
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    neuro_vals = [0.9662, 0.9623, 0.9658, 0.9590]
    svm_vals = [0.7845, 0.7723, 0.7891, 0.7806]
    rf_vals = [0.8432, 0.8367, 0.8498, 0.8431]
    gbm_vals = [0.8791, 0.8734, 0.8845, 0.8789]
    xgb_vals = [0.9134, 0.9087, 0.9178, 0.9132]
    ada_vals = [0.8567, 0.8523, 0.8612, 0.8567]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.1
    ax.bar(x - 0.25, neuro_vals, width, label='NeuroDetect-X', color='blue')
    ax.bar(x - 0.15, svm_vals, width, label='SVM', color='red')
    ax.bar(x - 0.05, rf_vals, width, label='RF', color='green')
    ax.bar(x + 0.05, gbm_vals, width, label='GBM', color='orange')
    ax.bar(x + 0.15, xgb_vals, width, label='XGBoost', color='purple')
    ax.bar(x + 0.25, ada_vals, width, label='AdaBoost', color='brown')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Score')
    ax.set_title('Figure 5: Model Performance Trends Across Accuracy, Precision, Recall, and F1-Score')
    ax.legend()
    plt.show()

plot_performance_trends()

# Figure 6: Comparative Analysis of DRS
def plot_drs_comparison():
    models = ['SVM', 'RF', 'GBM', 'XGBoost', 'AdaBoost', 'NeuroDetect-X']
    drs_vals = [0.7812, 0.8415, 0.8768, 0.9108, 0.8542, 0.9608]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(models, drs_vals, color='skyblue')
    ax.set_ylabel('DRS')
    ax.set_title('Figure 6: Comparative Analysis of Detection Rate Scores Across Model')
    plt.show()

plot_drs_comparison()

# Figure 7: Ablation Study
def plot_ablation():
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(configs))
    width = 0.25
    ax.bar(x - width, accuracies_ab, width, label='Accuracy', color='blue')
    ax.bar(x, f1_scores_ab, width, label='F1-Score', color='green')
    ax.bar(x + width, drs_ab, width, label='DRS', color='red')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45)
    ax.set_ylabel('Score')
    ax.set_title('Figure 7: Ablation Study: Accuracy, F1-Score, and DRS Across Different Configurations')
    ax.legend()
    plt.show()

plot_ablation()

# Figure 8: Feature Distribution
def plot_feature_distribution():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(counts, labels=categories, autopct='%1.1f%%', startangle=140)
    ax.set_title('Figure 8: Distribution of Selected Features by Category (Boruta)')
    plt.show()

plot_feature_distribution()

# Figure 9: Training Time Comparison
def plot_training_times():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(models_time, times, color='lightcoral')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Figure 9: Training Time Comparison Across Models')
    plt.show()

plot_training_times()

# Figure 10: Cross-Dataset Validation
def plot_cross_dataset():
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(datasets))
    width = 0.25
    ax.bar(x - width, acc_cross, width, label='Accuracy', color='blue')
    ax.bar(x, f1_cross, width, label='F1-Score', color='green')
    ax.bar(x + width, drs_cross, width, label='DRS', color='red')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel('Score')
    ax.set_title('Figure 10: Cross-Dataset Validation Performance on Primary vs. Independent Dataset')
    ax.legend()
    plt.show()

plot_cross_dataset()

# Figure 11: Early Detection Capabilities
def plot_early_detection():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(variants, accuracies_ed, color='lightgreen')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Figure 11: Early Detection Capabilities Across Parkinson’s Variants')
    plt.show()

plot_early_detection()

# Run all plots
print("All visualizations generated. Check the plots above.")
