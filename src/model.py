from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        class_weight={0:1, 1:5},
        random_state=42
    )

    model.fit(X_train, y_train)

    return model