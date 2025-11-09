import time
from ast import literal_eval
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from fma.plain import console, with_status
from fma.types import (
    DataFrame,
    EchonestDF,
    EchonestMetadataDF,
    FeaturesDF,
    Genre,
    GenreCSVDF,
    Genres,
    IDtoGenre,
    LibrosaDF,
    LibrosaLiteral,
    PathLike,
    TracksDF,
)

librosa_features_descriptions: dict[LibrosaLiteral, str] = {
    "chroma_cens": "Chroma Energy Normalized Statistics",
    "chroma_cqt": "Chroma with Constant-Q Transform",
    "chroma_stft": "Chroma with Short-Time Fourier Transform",
    "mfcc": "Mel-Frequency Cepstral Coefficients",
    "rmse": "Root Mean Square Energy",
    "spectral_bandwidth": "Spectral bandwidth",
    "spectral_centroid": "Spectral centroid",
    "spectral_contrast": "Spectral contrast",
    "spectral_rolloff": "Spectral rolloff",
    "tonnetz": "Tonnetz",
    "zcr": "Zero-Crossing Rate",
}

CSV_READER_CONFIG = {
    "tracks": {
        "chunksize": 5000,
        "num_chunks": 22,
    },
    "features": {
        "chunksize": 5000,
        "num_chunks": 22,
    },
    "echonest": {
        "chunksize": 5000,
        "num_chunks": 3,
    },
}


def _read_csv_pb(
    filepath: PathLike,
    chunksize: int = 5000,
    num_chunks: int | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Read a CSV file with a progress bar using chunked reading."""
    filepath = Path(filepath)

    # Read CSV in chunks with progress bar
    chunks = []
    reader = pd.read_csv(filepath, chunksize=chunksize, **kwargs)

    for chunk in console.track(reader, desc=f"Reading {filepath}", total=num_chunks):
        chunks.append(chunk)

    # Concatenate all chunks
    df = pd.concat(chunks, ignore_index=False)
    return df


@with_status
def read_features(path: PathLike, *, verbose: bool = True) -> LibrosaDF:
    if verbose:
        df = _read_csv_pb(
            path,
            header=[0, 1, 2],
            skiprows=[3],
            index_col=0,
            **CSV_READER_CONFIG["features"],
        )
    else:
        df = pd.read_csv(path, header=[0, 1, 2], skiprows=[3], index_col=0)
    df.columns = list(
        (
            feature,
            statistics,
            int(number),
        )
        for feature, statistics, number in zip(
            df.columns.get_level_values(0),
            df.columns.get_level_values(1),
            df.columns.get_level_values(2),
        )
    )
    df = cast(LibrosaDF, df)
    return df


@with_status
def read_echonest(
    path: PathLike, *, verbose: bool = True
) -> tuple[EchonestDF, EchonestMetadataDF]:
    if verbose:
        df = _read_csv_pb(
            path,
            header=[1, 2],
            skiprows=[3],
            index_col=0,
            **CSV_READER_CONFIG["echonest"],
        )
    else:
        df = pd.read_csv(path, header=[1, 2], skiprows=[3], index_col=0)
    df.columns = list(
        (
            catagory,
            entry,
        )
        for catagory, entry in zip(
            df.columns.get_level_values(0),
            df.columns.get_level_values(1),
        )
    )
    feature_df = df[
        [c for c in df.columns if c[0] not in ("metadata", "ranks", "social_features")]
    ]
    feature_df = cast(EchonestDF, feature_df)
    metadata_df = df[
        [c for c in df.columns if c[0] in ("metadata", "ranks", "social_features")]
    ]
    metadata_df = cast(EchonestMetadataDF, metadata_df)
    return feature_df, metadata_df


@with_status
def read_tracks(path: PathLike, *, verbose: bool = True) -> tuple[TracksDF, Genres]:
    if verbose:
        df = _read_csv_pb(
            path,
            header=[0, 1],
            skiprows=[2],
            index_col=0,
            **CSV_READER_CONFIG["tracks"],
        )
    else:
        df = pd.read_csv(path, header=[0, 1], skiprows=[2], index_col=0)

    # Define the track attributes we want
    track_attributes = [
        "bit_rate",
        "comments",
        "duration",
        "favorites",
        "interest",
        "listens",
    ]

    # Filter columns where level 0 is "track" and level 1 is in track_attributes
    track_cols = [
        col for col in df.columns if col[0] == "track" and col[1] in track_attributes
    ]
    track_df = df[track_cols]

    # Rename columns to just use level 1 (remove the "track" prefix)
    track_df.columns = [col[1] for col in track_cols]
    track_attribute_df = cast(TracksDF, track_df)

    # Get the genres_all column and convert string representations to lists
    genres_series = df[("track", "genres_all")].apply(literal_eval)
    track_genre = cast(Genres, genres_series)

    return track_attribute_df, track_genre


@with_status
def read_genres(path: PathLike, *, verbose: bool = True) -> IDtoGenre:
    df = pd.read_csv(path, header=0, index_col=0)
    df = cast(GenreCSVDF, df)
    id_to_genre: IDtoGenre = dict()
    for index, row in df.iterrows():
        parent_id = int(row["parent"])
        id_to_genre[index] = Genre(
            id=index,
            title=str(row["title"]),
            parent_id=parent_id if parent_id else None,
            top_level_id=int(row["top_level"]),
            children=list(),
        )
    for genre in id_to_genre.values():
        if genre.parent_id is not None:
            parent = id_to_genre[genre.parent_id]
            parent.children.append(genre.id)

    return id_to_genre


@dataclass
class FMADataset:
    librosa: LibrosaDF
    echonest: EchonestDF
    echonest_metadata: EchonestMetadataDF
    features: FeaturesDF
    tracks: TracksDF
    track_genres: Genres
    id_to_genre: IDtoGenre

    def assert_all_parent_genres_included(self) -> None:
        for i, g_list in enumerate(self.track_genres):
            for g in g_list:
                parent = self.id_to_genre[g].parent_id
                while parent is not None:
                    assert parent in g_list, (
                        f"Parent {parent} of genre {g} is not in {g_list} at index {i}"
                    )
                    parent = self.id_to_genre[parent].parent_id

    def remove_rare_genres(self, divider: int = 1000) -> None:
        counter = Counter(g for g_list in self.track_genres for g in g_list)
        n_threshold = len(self.track_genres) // divider

        # Filter out genres with counts less than threshold
        self.track_genres = self.track_genres.apply(
            lambda g_list: [g for g in g_list if counter[g] >= n_threshold]
        )

    @property
    def genre_ids(self) -> list[int]:
        return list(set(g for g_list in self.track_genres for g in g_list))

    @property
    def root_genre_ids(self) -> list[int]:
        return list(
            set(
                self.id_to_genre[g].top_level_id
                for g_list in self.track_genres
                for g in g_list
            )
        )

    @property
    def non_root_genre_ids(self) -> list[int]:
        return list(set(self.genre_ids) - set(self.root_genre_ids))

    def get_binary_labels(self, genre_id: int) -> list[bool]:
        return [genre_id in g_list for g_list in self.track_genres]

    @with_status(transient=False)
    def prepare_train_test(
        self,
        genre_set: Literal["all", "root", "non-root"] | Iterable[int] = "all",
        test_size: float = 0.2,
        random_state: int = 42,
        *,
        verbose: bool = True,
    ) -> tuple[
        FeaturesDF,
        FeaturesDF,
        DataFrame[int, int, bool],
        DataFrame[int, int, bool],
        StandardScaler,
    ]:
        if genre_set == "all":
            genre_set = self.genre_ids
        elif genre_set == "root":
            genre_set = self.root_genre_ids
        elif genre_set == "non-root":
            genre_set = self.non_root_genre_ids
        else:
            genre_set = list(genre_set)
            assert all(isinstance(genre_id, int) for genre_id in genre_set), (
                "genre_ids must be an iterable of integers, found non-integer values"
            )

        X = self.features
        per_genre_y = [self.get_binary_labels(genre_id) for genre_id in genre_set]
        Y = pd.DataFrame(
            np.stack(per_genre_y, axis=1),
            columns=genre_set,
            index=X.index,
        )
        Y = cast(DataFrame[int, int, bool], Y)
        train_idx, test_idx = (
            MultilabelStratifiedShuffleSplit(
                random_state=random_state,
                test_size=test_size,  # type: ignore
            )
            .split(X=X, y=Y)
            .__next__()
        )
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, Y_train, Y_test, scaler  # type: ignore


@with_status
def read_dataset(
    path: PathLike, cache: bool = True, *, verbose: bool = True
) -> FMADataset:
    path = Path(path).absolute()
    cache_dir = (path / ".cache").absolute()

    # Define cache file paths
    cache_files = {
        "tracks": cache_dir / "tracks.parquet",
        "genres": cache_dir / "genres.parquet",
        "echonest": cache_dir / "echonest.parquet",
        "echonest_metadata": cache_dir / "echonest_metadata.parquet",
        "librosa": cache_dir / "librosa.parquet",
        # No need to cache id_to_genre because it's a small dataset
    }

    # CSV source files
    csv_files = [
        path / "tracks.csv",
        path / "echonest.csv",
        path / "features.csv",
        path / "genres.csv",
    ]

    # Check if cache exists and is valid
    use_cache = False
    if cache and cache_dir.exists():
        all_cache_exist = all(f.exists() for f in cache_files.values())
        if all_cache_exist:
            # Check if cache is newer than all CSV files
            cache_mtime = min(f.stat().st_mtime for f in cache_files.values())
            csv_mtime = max(f.stat().st_mtime for f in csv_files if f.exists())
            use_cache = cache_mtime > csv_mtime

    if use_cache:
        # Load from cache
        start_time = time.time()

        status = console.status(
            f"Loading dataset from {cache_dir}", disable=not verbose
        )
        status.start()

        status.update(f"Loading tracks from {cache_files['tracks']}")
        tracks_df = pd.read_parquet(cache_files["tracks"])
        tracks_df = cast(TracksDF, tracks_df)

        status.update(f"Loading genres from {cache_files['genres']}")
        genres_series = pd.read_parquet(cache_files["genres"])
        genres = cast(Genres, genres_series.iloc[:, 0])

        status.update(f"Loading echonest features from {cache_files['echonest']}")
        echonest_df = pd.read_parquet(cache_files["echonest"])
        # Convert MultiIndex columns back to tuples
        echonest_df.columns = list(echonest_df.columns)
        echonest_df = cast(EchonestDF, echonest_df)

        status.update(
            f"Loading echonest metadata from {cache_files['echonest_metadata']}"
        )
        echonest_metadata_df = pd.read_parquet(cache_files["echonest_metadata"])
        # Convert MultiIndex columns back to tuples
        echonest_metadata_df.columns = list(echonest_metadata_df.columns)
        echonest_metadata_df = cast(EchonestMetadataDF, echonest_metadata_df)

        status.update(f"Loading librosa features from {cache_files['librosa']}")
        librosa_df = pd.read_parquet(cache_files["librosa"])
        # Convert MultiIndex columns back to tuples
        librosa_df.columns = list(librosa_df.columns)
        librosa_df = cast(LibrosaDF, librosa_df)

        status.update(f"Loading id-to-genre mapping from {path / 'genres.csv'}")
        id_to_genre = read_genres(path / "genres.csv")

        features_df = cast(FeaturesDF, pd.concat([librosa_df, echonest_df], axis=1))

        status.stop()
        unique_genres = set(chain(*genres.values))
        if verbose:
            console.done(
                f"Loaded {len(genres)} samples, "
                f"{features_df.shape[1]} features and "
                f"{len(unique_genres)} genres from cache in "
                f"{(time.time() - start_time) * 1000:.2f} ms"
            )
    else:
        # Read from CSV files
        if verbose and cache:
            console.info(f"Cache not found or outdated. Reading CSV files from {path}")
        start_time = time.time()
        tracks_df, genres = read_tracks(path / "tracks.csv", verbose=verbose)
        echonest_df, echonest_metadata_df = read_echonest(
            path / "echonest.csv", verbose=verbose
        )
        librosa_df = read_features(path / "features.csv", verbose=verbose)
        id_to_genre = read_genres(path / "genres.csv", verbose=verbose)
        librosa_df = librosa_df.loc[echonest_df.index]
        tracks_df = tracks_df.loc[echonest_df.index]
        genres = genres.loc[echonest_df.index]
        features_df = cast(
            FeaturesDF,
            pd.concat(
                [
                    librosa_df,
                    echonest_df,
                ],
                axis=1,
            ),
        )

        if verbose:
            unique_genres = set(chain(*genres.values))
            console.done(
                f"Loaded {len(genres)} samples, "
                f"{features_df.shape[1]} features and "
                f"{len(unique_genres)} genres in {(time.time() - start_time):.2f} seconds"
            )

        # Save to cache if requested
        if cache:
            start_time = time.time()
            cache_status = console.status(
                f"Saving dataset to {cache_dir}", disable=not verbose
            )
            cache_status.start()

            cache_dir.mkdir(exist_ok=True)

            cache_status.update(f"Saving tracks to {cache_files['tracks']}")
            tracks_df.to_parquet(cache_files["tracks"])
            cache_status.update(f"Saving genres to {cache_files['genres']}")
            genres.to_frame(name="genres").to_parquet(cache_files["genres"])
            cache_status.update(
                f"Saving echonest features to {cache_files['echonest']}"
            )

            # Convert tuple columns to MultiIndex for proper Parquet serialization
            echonest_df_to_save = echonest_df.copy()
            echonest_df_to_save.columns = pd.MultiIndex.from_tuples(echonest_df.columns)
            echonest_df_to_save.to_parquet(cache_files["echonest"])
            cache_status.update(
                f"Saving echonest metadata to {cache_files['echonest_metadata']}"
            )

            echonest_metadata_df_to_save = echonest_metadata_df.copy()
            echonest_metadata_df_to_save.columns = pd.MultiIndex.from_tuples(
                echonest_metadata_df.columns
            )
            echonest_metadata_df_to_save.to_parquet(cache_files["echonest_metadata"])
            cache_status.update(f"Saving librosa features to {cache_files['librosa']}")

            librosa_df_to_save = librosa_df.copy()
            librosa_df_to_save.columns = pd.MultiIndex.from_tuples(librosa_df.columns)
            librosa_df_to_save.to_parquet(cache_files["librosa"])

            cache_status.stop()
            if verbose:
                console.done(
                    f"Cached dataset to {cache_dir} in "
                    f"{time.time() - start_time:.2f} seconds"
                )

    dataset = FMADataset(
        track_genres=genres,
        features=features_df,
        id_to_genre=id_to_genre,
        tracks=tracks_df,
        librosa=librosa_df,
        echonest=echonest_df,
        echonest_metadata=echonest_metadata_df,
    )
    return dataset
