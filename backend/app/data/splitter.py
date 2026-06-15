from sklearn.model_selection import train_test_split
import numpy as np

def stratify_split(file_paths, labels, test_size=0.2, val_size=0.5):
    # Splits 80/10/10 preserving class balance
    X_train, X_temp, y_train, y_temp = train_test_split(
        file_paths, labels, test_size=test_size, stratify=labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
