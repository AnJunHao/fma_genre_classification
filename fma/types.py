from os import PathLike as PathLikeABC
from typing import Literal, overload, Any, TYPE_CHECKING
from collections.abc import Iterable, Callable, Iterator
import pandas as pd
from dataclasses import dataclass
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pandas.core.frame import _LocIndexerFrame, _iLocIndexerFrame
    from pandas.core.series import _LocIndexerSeries, _iLocIndexerSeries
else:

    class _LocIndexerFrame: ...

    class _iLocIndexerFrame: ...

    class _LocIndexerSeries: ...

    class _iLocIndexerSeries: ...


type Array1D[T] = NDArray
type Array2D[T] = NDArray


class LocIndexerSeries[T_index: Any, T_data: Any](_LocIndexerSeries):
    @overload
    def __getitem__(self, key: T_index) -> T_data: ...
    @overload
    def __getitem__(
        self, key: slice | Iterable[T_index]
    ) -> Series[T_index, T_data]: ...
    def __getitem__(
        self, key: T_index | slice | Iterable[T_index]
    ) -> T_data | Series[T_index, T_data]:
        return super().__getitem__(key)  # type: ignore


class iLocIndexerSeries[T_index: Any, T_data: Any](_iLocIndexerSeries):
    @overload
    def __getitem__(self, key: int) -> T_data: ...
    @overload
    def __getitem__(self, key: slice | Iterable[int]) -> Series[T_index, T_data]: ...
    def __getitem__(
        self, key: int | slice | Iterable[int]
    ) -> T_data | Series[T_index, T_data]:
        return super().__getitem__(key)  # type: ignore


class Series[T_index: Any, T_data: Any](pd.Series):
    loc: LocIndexerSeries[T_index, T_data]
    iloc: iLocIndexerSeries[T_index, T_data]
    values: Iterable[T_data]

    @overload
    def __getitem__(self, key: T_index) -> T_data: ...
    @overload
    def __getitem__(
        self, key: slice | Iterable[T_index]
    ) -> Series[T_index, T_data]: ...
    def __getitem__(
        self, key: T_index | slice | Iterable[T_index]
    ) -> T_data | Series[T_index, T_data]:
        return super().__getitem__(key)  # type: ignore

    def __iter__(self) -> Iterator[T_data]:
        return super().__iter__()

    @overload
    def apply(
        self, func: Callable[[T_data], T_data], *args: Any, **kwargs: Any
    ) -> Series[T_index, T_data]: ...
    @overload
    def apply(
        self, func: Callable[[T_data], Series], *args: Any, **kwargs: Any
    ) -> DataFrame: ...
    def apply(
        self, func: Callable[[T_data], T_data | Series], *args: Any, **kwargs: Any
    ) -> Series[T_index, T_data] | DataFrame:
        return super().apply(func, *args, **kwargs)  # type: ignore


class LocIndexerFrame[T_col: Any, T_index: Any, T_data: Any](_LocIndexerFrame):
    @overload
    def __getitem__(self, key: T_index) -> Series[T_col, T_data]: ...
    @overload
    def __getitem__(
        self, key: Iterable[T_index] | slice
    ) -> DataFrame[T_col, T_index, T_data]: ...
    def __getitem__(
        self, key: T_index | Iterable[T_index] | slice
    ) -> Series[T_col, T_data] | DataFrame[T_col, T_index, T_data]:
        return super().__getitem__(key)  # type: ignore


class ILocIndexerFrame[T_col: Any, T_index: Any, T_data: Any](_iLocIndexerFrame):
    @overload
    def __getitem__(self, key: int) -> Series[T_col, T_data]: ...
    @overload
    def __getitem__(
        self, key: Iterable[int] | slice
    ) -> DataFrame[T_col, T_index, T_data]: ...
    def __getitem__(
        self, key: int | Iterable[int] | slice
    ) -> DataFrame[T_col, T_index, T_data] | Series[T_col, T_data]:
        return super().__getitem__(key)  # type: ignore


class DataFrame[T_col: Any, T_index: Any, T_data: Any](pd.DataFrame):
    columns: pd.Index[T_col]
    index: pd.Index[T_index]
    loc: LocIndexerFrame[T_col, T_index, T_data]
    iloc: ILocIndexerFrame[T_col, T_index, T_data]

    @overload
    def __getitem__(
        self, key: Iterable[T_col] | slice
    ) -> DataFrame[T_col, T_index, T_data]: ...
    @overload
    def __getitem__(self, key: T_col) -> Series[T_index, T_data]: ...
    def __getitem__(
        self, key: T_col | Iterable[T_col] | slice
    ) -> DataFrame[T_col, T_index, T_data] | Series[T_index, T_data]:
        return super().__getitem__(key)

    def iterrows(self) -> Iterator[tuple[T_index, Series[T_col, T_data]]]:
        return super().iterrows()  # type: ignore


type PathLike = PathLikeABC | str

type LibrosaLiteral = Literal[
    "chroma_cens",
    "chroma_cqt",
    "chroma_stft",
    "mfcc",
    "rmse",
    "spectral_bandwidth",
    "spectral_centroid",
    "spectral_contrast",
    "spectral_rolloff",
    "tonnetz",
    "zcr",
]
type StatisticsLiteral = Literal[
    "kurtosis", "max", "mean", "median", "min", "skew", "std"
]
type LibrosaCol = tuple[LibrosaLiteral, StatisticsLiteral, int]
type LibrosaDF = DataFrame[LibrosaCol, int, float]

type EchonestCategory = Literal["audio_features", "temporal_features"]
type EchonestCol = tuple[EchonestCategory, str]
type EchonestDF = DataFrame[EchonestCol, int, float]
type EchonestMetadataCategory = Literal["ranks", "metadata", "social_features"]
type EchonestMetadataCol = tuple[EchonestMetadataCategory, str]
type EchonestMetadataDF = DataFrame[EchonestMetadataCol, int, object]

type TracksCol = Literal[
    "bit_rate", "comments", "duration", "favorites", "interest", "listens"
]
type TracksDF = DataFrame[TracksCol, int, int]
type Genres = Series[int, list[int]]


@dataclass
class Genre:
    id: int
    title: str
    parent_id: int | None
    top_level_id: int
    children: list[int]


type IDtoGenre = dict[int, Genre]
type GenreCSVCol = Literal["genre_id", "#tracks", "parent", "title", "top_level"]
type GenreCSVDF = DataFrame[GenreCSVCol, int, int | str]

type FeaturesCol = LibrosaCol | EchonestCol
type FeaturesDF = DataFrame[FeaturesCol, int, float]
