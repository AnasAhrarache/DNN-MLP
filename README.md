# Deep Learning Lab 01 — Regression and Multiclass Classification

**Institution:** Université Abdelmalek Essaâdi — FST Tanger  
**Program:** Master MBD  
**Instructor:** Pr. ELAACHAK Lotfi

## Project overview

This lab explores the practical use of **PyTorch** to solve two different supervised learning problems. The first notebook estimates stock closing prices through regression; the second identifies machine failure categories through multiclass classification. Both exercises cover data exploration, feature preparation, neural network training, hyperparameter selection, and evaluation.

## Repository structure

| Notebook | Focus | Dataset |
| --- | --- | --- |
| `Part1_Regression_NYSE.ipynb` | Predicting stock closing prices | [NYSE dataset](https://www.kaggle.com/datasets/dgawlik/nyse) |
| `Part2_Classification_Maintenance.ipynb` | Predictive maintenance failure classification | [Machine Predictive Maintenance](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) |

The notebooks use **`kagglehub`** to retrieve their datasets automatically. They are intended to run in notebook environments such as Google Colab or Kaggle, with the required Python dependencies available.

## Exercise A — Stock price regression

**Goal:** Estimate the `close` stock price using the `open`, `low`, `high`, and `volume` attributes.

**Workflow:**
1. Examine dataset dimensions, missing values, feature distributions, correlations, price trends, and trading volumes.
2. Transform trading volume using a logarithmic transformation and standardize input features.
3. Train a configurable multilayer perceptron (MLP) built from linear layers and ReLU activations.
4. Tune learning rate, optimizer (Adam or SGD), epochs, hidden-layer width, and network depth using `skorch` and `GridSearchCV`.
5. Compare training and testing trends using loss and R² curves across epochs.
6. Investigate dropout, L1, L2/weight decay, batch normalization, and combined regularization.

**Key consideration:** Strong relationships between stock price variables can make this regression task relatively easy; added regularization does not automatically improve generalization.

## Exercise B — Predictive maintenance classification

**Goal:** Assign equipment observations to one of five remaining failure categories after data cleaning.

**Workflow:**
1. Inspect the original class proportions, distributions, correlations, and feature behavior by class.
2. Remove identifiers and contradictory `Target`/`Failure Type` records; exclude `Target` to prevent target leakage.
3. Encode categorical inputs and labels, then standardize numerical inputs.
4. Apply SMOTE **to the training set only** to address class imbalance.
5. Train an MLP with five output units and cross-entropy loss.
6. Select hyperparameters with `skorch`/`GridSearchCV`, using **macro F1** as the search metric.
7. Review epoch-by-epoch training/testing loss and accuracy, as well as precision, recall (sensitivity), specificity, F1 (macro and weighted), and confusion matrices.
8. Compare the unregularized model with dropout, L1, L2, batch normalization, and combined variants.

**Key consideration:** The original dataset is heavily dominated by the *No Failure* category, so accuracy alone is not a sufficient measure of classification quality. Per-class recall and macro F1 provide important additional context.

## Main concepts practiced

- Defining and training neural networks with `torch.nn.Module`, forward passes, backpropagation, and optimizer updates.
- Connecting PyTorch models with scikit-learn workflows using `skorch`.
- Selecting architectures and training settings through systematic grid search.
- Recognizing and avoiding target leakage during preprocessing.
- Managing imbalanced classes while keeping the test set separate from synthetic oversampling.
- Reading learning curves to assess underfitting, overfitting, and generalization.
- Comparing regularization techniques based on task-specific validation and test metrics.

## Running the notebooks

1. Open either `.ipynb` file in Google Colab, Kaggle, or a compatible local Jupyter environment.
2. Execute its cells in order. Install any missing packages requested by the notebook.
3. Allow `kagglehub` to fetch the relevant dataset, then inspect the produced figures, metrics, and model comparisons.

> **Note:** This version updates the project documentation only. The two original notebooks and their code, outputs, and datasets references have not been changed.
