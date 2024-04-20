import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import math
from zipfile import ZipFile
from urllib.request import urlretrieve

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.layers import StringLookup


EPOCHS = 10
SEQ_LEN = 4
STEP_SIZE = 2
USER_FEATURES = None
CATEGORICAL_FEATURES_WITH_VOCABULARY = None
CSV_HEADER = None
movies = None
genres = None


# Download the dataset
def download():
    urlretrieve(
        "http://files.grouplens.org/datasets/movielens/ml-1m.zip", "movielens.zip"
    )
    ZipFile("movielens.zip", "r").extractall()


def load_data():
    users = pd.read_csv(
        "ml-1m/users.dat",
        sep="::",
        names=["user_id", "sex", "age_group", "occupation", "zip_code"],
        encoding="ISO-8859-1",
        engine="python",
    )

    ratings = pd.read_csv(
        "ml-1m/ratings.dat",
        sep="::",
        names=["user_id", "movie_id", "rating", "unix_timestamp"],
        encoding="ISO-8859-1",
        engine="python",
    )

    movies = pd.read_csv(
        "ml-1m/movies.dat",
        sep="::",
        names=["movie_id", "title", "genres"],
        encoding="ISO-8859-1",
        engine="python",
    )

    return users, ratings, movies


def create_sequences(values, window_size, step_size):
    sequences = []
    start_index = 0
    while True:
        end_index = start_index + window_size
        seq = values[start_index:end_index]
        if len(seq) < window_size:
            seq = values[-window_size:]
            if len(seq) == window_size:
                sequences.append(seq)
            break
        sequences.append(seq)
        start_index += step_size
    return sequences


def get_dataset_from_csv(csv_file_path, shuffle=False, batch_size=128):
    def process(features):
        movie_ids_string = features["sequence_movie_ids"]
        sequence_movie_ids = tf.strings.split(movie_ids_string, ",").to_tensor()

        # The last movie id in the sequence is the target movie.
        features["target_movie_id"] = sequence_movie_ids[:, -1]
        features["sequence_movie_ids"] = sequence_movie_ids[:, :-1]

        ratings_string = features["sequence_ratings"]
        sequence_ratings = tf.strings.to_number(
            tf.strings.split(ratings_string, ","), tf.dtypes.float32
        ).to_tensor()

        # The last rating in the sequence is the target for the model to predict.
        target = sequence_ratings[:, -1]
        features["sequence_ratings"] = sequence_ratings[:, :-1]

        return features, target

    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=CSV_HEADER,
        num_epochs=1,
        header=False,
        field_delim="|",
        shuffle=shuffle,
    ).map(process)

    return dataset


def create_model_inputs():
    return {
        "user_id": keras.Input(name="user_id", shape=(1,), dtype="string"),
        "sequence_movie_ids": keras.Input(
            name="sequence_movie_ids", shape=(SEQ_LEN - 1,), dtype="string"
        ),
        "target_movie_id": keras.Input(
            name="target_movie_id", shape=(1,), dtype="string"
        ),
        "sequence_ratings": keras.Input(
            name="sequence_ratings", shape=(SEQ_LEN - 1,), dtype=tf.float32
        ),
        "sex": keras.Input(name="sex", shape=(1,), dtype="string"),
        "age_group": keras.Input(name="age_group", shape=(1,), dtype="string"),
        "occupation": keras.Input(name="occupation", shape=(1,), dtype="string"),
    }


def encode_input_features(
    inputs,
    include_user_id=True,
    include_user_features=True,
    include_movie_features=True,
):
    encoded_transformer_features = []
    encoded_other_features = []

    other_feature_names = []
    if include_user_id:
        other_feature_names.append("user_id")
    if include_user_features:
        other_feature_names.extend(USER_FEATURES)

    ## Encode user features
    for feature_name in other_feature_names:
        # Convert the string input values into integer indices.
        vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
        idx = StringLookup(vocabulary=vocabulary, mask_token=None, num_oov_indices=0)(
            inputs[feature_name]
        )
        # Compute embedding dimensions
        embedding_dims = int(math.sqrt(len(vocabulary)))
        # Create an embedding layer with the specified dimensions.
        embedding_encoder = layers.Embedding(
            input_dim=len(vocabulary),
            output_dim=embedding_dims,
            name=f"{feature_name}_embedding",
        )
        # Convert the index values to embedding representations.
        encoded_other_features.append(embedding_encoder(idx))

    ## Create a single embedding vector for the user features
    if len(encoded_other_features) > 1:
        encoded_other_features = layers.concatenate(encoded_other_features)
    elif len(encoded_other_features) == 1:
        encoded_other_features = encoded_other_features[0]
    else:
        encoded_other_features = None

    ## Create a movie embedding encoder
    movie_vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY["movie_id"]
    movie_embedding_dims = int(math.sqrt(len(movie_vocabulary)))
    # Create a lookup to convert string values to integer indices.
    movie_index_lookup = StringLookup(
        vocabulary=movie_vocabulary,
        mask_token=None,
        num_oov_indices=0,
        name="movie_index_lookup",
    )
    # Create an embedding layer with the specified dimensions.
    movie_embedding_encoder = layers.Embedding(
        input_dim=len(movie_vocabulary),
        output_dim=movie_embedding_dims,
        name=f"movie_embedding",
    )
    # Create a vector lookup for movie genres.
    genre_vectors = movies[genres].to_numpy()
    movie_genres_lookup = layers.Embedding(
        input_dim=genre_vectors.shape[0],
        output_dim=genre_vectors.shape[1],
        embeddings_initializer=keras.initializers.Constant(genre_vectors),
        trainable=False,
        name="genres_vector",
    )
    # Create a processing layer for genres.
    movie_embedding_processor = layers.Dense(
        units=movie_embedding_dims,
        activation="relu",
        name="process_movie_embedding_with_genres",
    )

    ## Define a function to encode a given movie id.
    def encode_movie(movie_id):
        # Convert the string input values into integer indices.
        movie_idx = movie_index_lookup(movie_id)
        movie_embedding = movie_embedding_encoder(movie_idx)
        encoded_movie = movie_embedding
        if include_movie_features:
            movie_genres_vector = movie_genres_lookup(movie_idx)
            encoded_movie = movie_embedding_processor(
                layers.concatenate([movie_embedding, movie_genres_vector])
            )
        return encoded_movie

    ## Encoding target_movie_id
    target_movie_id = inputs["target_movie_id"]
    encoded_target_movie = encode_movie(target_movie_id)

    ## Encoding sequence movie_ids.
    sequence_movies_ids = inputs["sequence_movie_ids"]
    encoded_sequence_movies = encode_movie(sequence_movies_ids)
    # Create positional embedding.
    position_embedding_encoder = layers.Embedding(
        input_dim=SEQ_LEN,
        output_dim=movie_embedding_dims,
        name="position_embedding",
    )
    positions = tf.range(start=0, limit=SEQ_LEN - 1, delta=1)
    encodded_positions = position_embedding_encoder(positions)
    # Retrieve sequence ratings to incorporate them into the encoding of the movie.
    sequence_ratings = inputs["sequence_ratings"]
    sequence_ratings = keras.ops.expand_dims(sequence_ratings, -1)
    # Add the positional encoding to the movie encodings and multiply them by rating.
    encoded_sequence_movies_with_poistion_and_rating = layers.Multiply()(
        [(encoded_sequence_movies + encodded_positions), sequence_ratings]
    )

    # Construct the transformer inputs.
    for i in range(SEQ_LEN - 1):
        feature = encoded_sequence_movies_with_poistion_and_rating[:, i, ...]
        feature = keras.ops.expand_dims(feature, 1)
        encoded_transformer_features.append(feature)
    encoded_transformer_features.append(encoded_target_movie)

    encoded_transformer_features = layers.concatenate(
        encoded_transformer_features, axis=1
    )

    return encoded_transformer_features, encoded_other_features


def create_model(
    num_heads,
    dropout_rate,
    hidden_units,
    include_user_id=True,
    include_user_features=True,
    include_movie_features=True,
):
    inputs = create_model_inputs()
    transformer_features, other_features = encode_input_features(
        inputs, include_user_id, include_user_features, include_movie_features
    )

    # Create a multi-headed attention layer.
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=transformer_features.shape[2], dropout=dropout_rate
    )(transformer_features, transformer_features)

    # Transformer block.
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    x1 = layers.Add()([transformer_features, attention_output])
    x1 = layers.LayerNormalization()(x1)
    x2 = layers.LeakyReLU()(x1)
    x2 = layers.Dense(units=x2.shape[-1])(x2)
    x2 = layers.Dropout(dropout_rate)(x2)
    transformer_features = layers.Add()([x1, x2])
    transformer_features = layers.LayerNormalization()(transformer_features)
    features = layers.Flatten()(transformer_features)

    # Included the other features.
    if other_features is not None:
        features = layers.concatenate(
            [features, layers.Reshape([other_features.shape[-1]])(other_features)]
        )

    # Fully-connected layers.
    for num_units in hidden_units:
        features = layers.Dense(num_units)(features)
        features = layers.BatchNormalization()(features)
        features = layers.LeakyReLU()(features)
        features = layers.Dropout(dropout_rate)(features)

    outputs = layers.Dense(units=1)(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def main():
    # download()
    global movies, genres, CATEGORICAL_FEATURES_WITH_VOCABULARY, USER_FEATURES, CSV_HEADER
    users, ratings, movies = load_data()
    # processing
    users["user_id"] = users["user_id"].apply(lambda x: f"user_{x}")
    users["age_group"] = users["age_group"].apply(lambda x: f"group_{x}")
    users["occupation"] = users["occupation"].apply(lambda x: f"occupation_{x}")

    movies["movie_id"] = movies["movie_id"].apply(lambda x: f"movie_{x}")

    ratings["movie_id"] = ratings["movie_id"].apply(lambda x: f"movie_{x}")
    ratings["user_id"] = ratings["user_id"].apply(lambda x: f"user_{x}")
    ratings["rating"] = ratings["rating"].apply(lambda x: float(x))

    genres = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime"]
    genres += ["Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical"]
    genres += ["Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

    # One-hot encoding for genres
    for genre in genres:
        movies[genre] = movies["genres"].apply(
            lambda values: int(genre in values.split("|"))
        )

    ratings_group = ratings.sort_values(by=["unix_timestamp"]).groupby("user_id")

    # df of each users movie ratings
    ratings_data = pd.DataFrame(
        data={
            "user_id": list(ratings_group.groups.keys()),
            "movie_ids": list(ratings_group.movie_id.apply(list)),
            "ratings": list(ratings_group.rating.apply(list)),
            "timestamps": list(ratings_group.unix_timestamp.apply(list)),
        }
    )

    ratings_data.movie_ids = ratings_data.movie_ids.apply(
        lambda ids: create_sequences(ids, SEQ_LEN, STEP_SIZE)
    )

    ratings_data.ratings = ratings_data.ratings.apply(
        lambda ids: create_sequences(ids, SEQ_LEN, STEP_SIZE)
    )

    del ratings_data["timestamps"]

    ratings_data_movies = ratings_data[["user_id", "movie_ids"]].explode(
        "movie_ids", ignore_index=True
    )

    ratings_data_rating = ratings_data[["ratings"]].explode(
        "ratings", ignore_index=True
    )

    # concat ratings and movie_ids
    ratings_data_transformed = pd.concat(
        [ratings_data_movies, ratings_data_rating], axis=1
    )

    ratings_data_transformed = ratings_data_transformed.join(
        users.set_index("user_id"), on="user_id"
    )

    ratings_data_transformed.movie_ids = ratings_data_transformed.movie_ids.apply(
        lambda x: ",".join(x)
    )

    ratings_data_transformed.ratings = ratings_data_transformed.ratings.apply(
        lambda x: ",".join([str(v) for v in x])
    )

    del ratings_data_transformed["zip_code"]

    ratings_data_transformed.rename(
        columns={"movie_ids": "sequence_movie_ids", "ratings": "sequence_ratings"},
        inplace=True,
    )

    random_selection = np.random.rand(len(ratings_data_transformed.index)) <= 0.85
    train_data = ratings_data_transformed[random_selection]
    test_data = ratings_data_transformed[~random_selection]

    train_data.to_csv("train_data.csv", index=False, sep="|", header=False)
    test_data.to_csv("test_data.csv", index=False, sep="|", header=False)

    CSV_HEADER = list(ratings_data_transformed.columns)

    CATEGORICAL_FEATURES_WITH_VOCABULARY = {
        "user_id": list(users.user_id.unique()),
        "movie_id": list(movies.movie_id.unique()),
        "sex": list(users.sex.unique()),
        "age_group": list(users.age_group.unique()),
        "occupation": list(users.occupation.unique()),
    }

    USER_FEATURES = ["sex", "age_group", "occupation"]

    MOVIE_FEATURES = ["genres"]

    print(CSV_HEADER)
    get_dataset_from_csv("train_data.csv", shuffle=True, batch_size=5)

    include_user_id = False
    include_user_features = False
    include_movie_features = False

    hidden_units = [256, 128]
    dropout_rate = 0.1
    num_heads = 3
    model = create_model(
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        hidden_units=hidden_units,
        include_user_id=include_user_id,
        include_user_features=include_user_features,
        include_movie_features=include_movie_features,
    )

    # Compile the model.
    model.compile(
        optimizer=keras.optimizers.Adagrad(learning_rate=0.01),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError()],
    )

    # Read the training data.
    train_dataset = get_dataset_from_csv("train_data.csv", shuffle=True, batch_size=265)

    # Fit the model with the training data.
    model.fit(train_dataset, epochs=EPOCHS)

    # Read the test data.
    test_dataset = get_dataset_from_csv("test_data.csv", batch_size=265)
    print(test_dataset.take(1))
    print(model.predict(test_dataset.take(1), verbose=1))

    # Evaluate the model on the test data.
    _, rmse = model.evaluate(test_dataset, verbose=0)
    print(f"Test MAE: {round(rmse, 3)}")


if __name__ == "__main__":
    main()
