import pandas as pd
import numpy as np


def load_and_clean_data():
    try:
        books   = pd.read_csv('Books.csv',   sep=None, engine='python', on_bad_lines='skip', encoding='latin-1')
        ratings = pd.read_csv('Ratings.csv', sep=None, engine='python', encoding='latin-1')
        users   = pd.read_csv('Users.csv',   sep=None, engine='python', encoding='latin-1')
    except Exception as e:
        print(f"File read error: {e}")
        return None

    # Standardize column names
    ratings.columns = ['User-ID', 'ISBN', 'Rating']
    books.columns   = ['ISBN', 'Title', 'Author', 'Year', 'Publisher']
    users.columns   = ['User-ID', 'Age']

    # Type alignment for merge
    ratings['User-ID'] = ratings['User-ID'].astype(str)
    users['User-ID']   = users['User-ID'].astype(str)

    # Keep only explicit ratings (0 = implicit)
    ratings = ratings[ratings['Rating'] > 0].copy()

    # Merge all three
    df = ratings.merge(books[['ISBN', 'Title', 'Author']], on='ISBN', how='left')
    df = df.merge(users, on='User-ID', how='left')
    df = df.dropna(subset=['ISBN', 'Rating'])
    df.columns = df.columns.str.strip()

    # Numeric indices
    df['user_idx'] = df['User-ID'].astype('category').cat.codes
    df['book_idx'] = df['ISBN'].astype('category').cat.codes

    # Normalize rating [1,10] → [0,1]
    df['rating_norm'] = (df['Rating'].astype(float) - 1) / 9.0

    # ── Statistical features ──────────────────────────────────────────────────
    # Per-user average rating
    user_mean  = df.groupby('user_idx')['rating_norm'].mean().rename('user_mean')
    # Per-book average rating
    book_mean  = df.groupby('book_idx')['rating_norm'].mean().rename('book_mean')
    # Activity counts (log-normalized)
    user_count = df.groupby('user_idx')['rating_norm'].count().rename('user_count')
    book_count = df.groupby('book_idx')['rating_norm'].count().rename('book_count')

    df = df.join(user_mean,  on='user_idx')
    df = df.join(book_mean,  on='book_idx')
    df = df.join(user_count, on='user_idx', rsuffix='_uc')
    df = df.join(book_count, on='book_idx', rsuffix='_bc')

    df['user_count_norm'] = np.log1p(df['user_count']) / np.log1p(df['user_count'].max())
    df['book_count_norm'] = np.log1p(df['book_count']) / np.log1p(df['book_count'].max())

    # ── Age feature ───────────────────────────────────────────────────────────
    # Clean age: valid range 5-100, fill missing with median
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df.loc[(df['Age'] < 5) | (df['Age'] > 100), 'Age'] = np.nan
    age_median = df['Age'].median()
    df['age_norm'] = df['Age'].fillna(age_median) / 100.0   # normalize to [0,1]

    print(f"[Preprocessing] Dataset loaded: {len(df):,} explicit ratings")
    print(f"[Preprocessing] Unique users: {df['user_idx'].nunique():,} | "
          f"Unique books: {df['book_idx'].nunique():,}")
    print(f"[Preprocessing] Age coverage: {df['Age'].notna().mean()*100:.1f}%  "
          f"(median fill = {age_median:.0f})")

    return df


def get_train_data(df, sample_size=5000):
    df = df.copy()
    df['rating_int'] = df['Rating'].astype(int)

    bins   = [0, 2, 4, 6, 8, 10]
    labels = ['1-2', '3-4', '5-6', '7-8', '9-10']
    df['group'] = pd.cut(df['rating_int'], bins=bins, labels=labels)

    base_quota    = sample_size // len(labels)
    group_weights = {'1-2': 0.8, '3-4': 0.8, '5-6': 1.0, '7-8': 1.2, '9-10': 1.2}

    parts = []
    for label in labels:
        g = df[df['group'] == label]
        if len(g) == 0:
            continue
        n = min(len(g), int(base_quota * group_weights[label]))
        parts.append(g.sample(n=n, random_state=42))

    subset = pd.concat(parts).sample(frac=1, random_state=42)

    print(f"\n[Preprocessing] Training set: {len(subset):,} samples")
    print(subset['group'].value_counts().sort_index().to_string())

    FEATURES = ['user_mean', 'book_mean', 'user_count_norm', 'book_count_norm', 'age_norm']
    X = subset[FEATURES].values
    y = subset['rating_norm'].values.reshape(-1, 1)

    return X, y