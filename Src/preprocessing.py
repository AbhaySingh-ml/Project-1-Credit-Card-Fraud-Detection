from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def scale_features(df, feature='Amount'):
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[[feature]])
    return df.drop(columns=[feature])

def split_data(df, target='Class', test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

