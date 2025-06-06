import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.preprocessing import OrdinalEncoder
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

def train_model(df: pd.DataFrame, model_type: str = 'catboost', target_col: str = 'type_case', threshold: float = 0.3):
    """
    Trains a classification model (CatBoost or XGBoost) on the given dataframe.

    Parameters:
        df (pd.DataFrame): The input DataFrame with features and a target column.
        model_type (str): 'catboost' or 'xgboost'.
        target_col (str): The name of the target column to predict.
        threshold (float): Probability threshold for binary classification.

    Returns:
        model: The trained model instance.
        df_pred (pd.DataFrame): DataFrame with predictions, probabilities, and ground truth.
    """
    print("Preparing data...")
    X = df.drop(columns=[target_col, 'sample_id'])
    y = df[target_col].values

    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    if model_type == 'catboost':
        print("Filling missing categorical values with placeholder for CatBoost...")
        for c in cat_cols:
            X[c] = X[c].astype(str).fillna('nan')

    elif model_type == 'xgboost':
        print("Encoding categorical columns with OrdinalEncoder for XGBoost...")
        le = LabelEncoder()
        y = le.fit_transform(y)
        if cat_cols:
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X[cat_cols] = encoder.fit_transform(X[cat_cols])

    else:
        raise ValueError("Unsupported model type. Choose 'catboost' or 'xgboost'.")

    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    if model_type == 'catboost':
        print("Training CatBoostClassifier...")
        train_pool = Pool(data=X_train, label=y_train, cat_features=cat_cols)
        test_pool = Pool(data=X_test, label=y_test, cat_features=cat_cols)
        model = CatBoostClassifier(
            iterations=300,
            learning_rate=0.01,
            depth=6,
            eval_metric='Recall',
            random_seed=42,
            verbose=False
        )
        model.fit(train_pool, eval_set=test_pool)
        y_pred = model.predict(test_pool)
        y_pred_prob = model.predict_proba(test_pool)

    elif model_type == 'xgboost':
        print("Training XGBoostClassifier...")
        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.01,
            max_depth=6,
            use_label_encoder=False,
            eval_metric='recall',
            random_state=42
        )
        model.fit(X_train, y_train)
        y_test = le.inverse_transform(y_test)
        y_pred = le.inverse_transform(model.predict(X_test))
        y_pred_prob = model.predict_proba(X_test)

    # Initial evaluation
    print("Evaluating model (threshold = 0.5)...")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='granted', zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label='granted', zero_division=0)
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print("\nFull classification report:\n", classification_report(y_test, y_pred))

    df_pred = pd.DataFrame()
    df_pred['preds'] = y_pred
    df_pred['preds_prob1'] = y_pred_prob[:, 1]
    df_pred['preds_prob0'] = y_pred_prob[:, 0]
    df_pred['gt_values'] = y_test

    # Insights on probabilities
    print("Median predicted probabilities:")
    print('Prob Refused of refused:  %.3f', df_pred[df_pred.gt_values == 'refused'].preds_prob1.median())
    print('Prob Granted of refused:  %.3f', df_pred[df_pred.gt_values == 'refused'].preds_prob0.median())
    print('Prob Refused of granted:  %.3f', df_pred[df_pred.gt_values == 'granted'].preds_prob1.median())
    print('Prob Granted of granted:  %.3f', df_pred[df_pred.gt_values == 'granted'].preds_prob0.median())

    # Thresholding
    pred_truc = np.where(y_pred_prob[:, 1] > threshold, 'refused', 'granted')

    # Re-evaluation with threshold
    print(f"Evaluating with custom threshold = {threshold}...")
    acc = accuracy_score(y_test, pred_truc)
    prec = precision_score(y_test, pred_truc, pos_label='granted', zero_division=0)
    rec = recall_score(y_test, pred_truc, pos_label='granted', zero_division=0)
    print(f"Thresholded Accuracy:  {acc:.3f}")
    print(f"Thresholded Precision: {prec:.3f}")
    print(f"Thresholded Recall:    {rec:.3f}")
    print("\nFull classification report (with threshold):\n", classification_report(y_test, pred_truc))

    # Confusion matrix
    labels = ['granted', 'refused']
    cm = confusion_matrix(y_test, pred_truc, labels=labels)
    cm_df = pd.DataFrame(cm,
                         index=[f"actual_{lab}" for lab in labels],
                         columns=[f"predicted_{lab}" for lab in labels])
    print("Confusion Matrix:\n%s", cm_df)

    return model, df_pred
