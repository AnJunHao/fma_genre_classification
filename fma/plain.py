import builtins
import re
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from functools import wraps
from itertools import repeat
from typing import Any, Literal, Pattern, TypedDict, cast, overload

import pandas as pd
from rich.console import Console
from rich.highlighter import Highlighter
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.status import Status
from rich.table import Table
from rich.text import Text


def format_value(value: Any, fmt_spec: str | None) -> str:
    """Format a given value using a string format specifier."""
    if fmt_spec is None:
        return str(value)
    try:
        return format(value, fmt_spec)
    except Exception:
        return str(value)


def determine_justify(fmt_spec: str) -> Literal["left", "right", "center"]:
    """Determine the justification for a column by guessing data type from format specifier."""
    fmt_spec = fmt_spec.strip()
    if ">" in fmt_spec:
        # Aligns right
        return "right"
    elif "<" in fmt_spec:
        # Aligns left
        return "left"
    elif "=" in fmt_spec or "^" in fmt_spec:
        # Aligns center
        return "center"
    elif any(c in fmt_spec for c in ["f", "%", "e", "E", "g", "G"]):
        # Data is likely float
        return "right"
    elif any(c in fmt_spec for c in ["d", "b", "x", "X"]):
        # Data is likely integer
        return "right"
    elif "s" in fmt_spec:
        # Data is likely string
        return "left"
    elif "." in fmt_spec or "," in fmt_spec:
        # Ambiguous, but likely numeric
        return "right"
    else:
        return "left"


class TextTheme(TypedDict):
    default: str | None
    number: str | None
    path: str | None
    first: str | None
    prefix: str | None
    prefix_style: str | None


class ProgressTheme(TypedDict):
    desc: TextTheme | SemanticHighlighter
    spinner: tuple[str, str, str]  # spinner type, spinner style, spinner finished text
    bar: tuple[
        str, str, str, str
    ]  # background style, complete style, finished style, pulse style
    percent: str


class StatusTheme(TypedDict):
    desc: TextTheme | SemanticHighlighter
    spinner: tuple[str, str]  # spinner type, spinner style


class ThemeMap(TypedDict):
    default: TextTheme | SemanticHighlighter
    done: TextTheme | SemanticHighlighter
    minor_done: TextTheme | SemanticHighlighter
    warn: TextTheme | SemanticHighlighter
    info: TextTheme | SemanticHighlighter
    progress: ProgressTheme
    status: StatusTheme


class SemanticHighlighter(Highlighter):
    """Highlighter that applies semantic highlighting using a theme."""

    def __init__(self, theme: TextTheme):
        """Initialize the highlighter with a theme.

        Args:
            theme: A TextTheme dictionary containing style definitions.
        """
        self.theme = theme
        self.patterns: list[tuple[Pattern[str], Literal["number", "path"]]] = [
            (re.compile(r"\b\d+(?:\.\d+)?\b"), "number"),
            (re.compile(r"(?:~|/|[A-Za-z]:\\)[A-Za-z0-9_\-./\\]+"), "path"),
        ]

    def __call__(self, text: str | Text) -> Text:
        """Highlight a str or Text instance.

        Args:
            text (Union[str, ~Text]): Text to highlight.

        Raises:
            TypeError: If not called with text or str.

        Returns:
            Text: A test instance with highlighting applied.
        """
        if isinstance(text, str):
            if self.theme["prefix"] is not None:
                highlight_text = Text(self.theme["prefix"] + text)
            else:
                highlight_text = Text(text)
        elif isinstance(text, Text):
            text = text.plain
            if self.theme["prefix"] is not None:
                highlight_text = Text(self.theme["prefix"] + text)
            else:
                highlight_text = Text(text)
        else:
            raise TypeError(f"str or Text instance required, not {text!r}")
        self.highlight(highlight_text)
        return highlight_text

    def highlight(self, text: Text) -> None:
        """Apply highlighting in place to text.

        Args:
            text: A Text object to highlight.
        """
        # Get the plain text to work with
        plain_text = text.plain

        # Calculate prefix offset if prefix exists
        prefix_len = len(self.theme["prefix"]) if self.theme["prefix"] else 0

        # Apply default style to entire text
        if self.theme["default"] is not None:
            text.stylize(self.theme["default"], 0, len(plain_text))

        # Style the prefix if present
        if self.theme["prefix"] is not None and self.theme["prefix_style"] is not None:
            text.stylize(self.theme["prefix_style"], 0, prefix_len)

        # Detect and style the first word (after prefix)
        if self.theme["first"] is not None:
            match = re.match(r"^(\w+)", plain_text[prefix_len:])
            if match:
                start, end = match.span(1)
                text.stylize(self.theme["first"], prefix_len + start, prefix_len + end)

        # Apply all regex-based styles
        for pattern, style_key in self.patterns:
            style = self.theme[style_key]
            if style is not None:
                for m in pattern.finditer(plain_text[prefix_len:]):
                    s, e = m.span()
                    text.stylize(style, prefix_len + s, prefix_len + e)


class CustomStatus(Status):
    def __init__(
        self,
        status: str,
        *args: Any,
        console: Console,
        highlighter: Highlighter,
        disable: bool = False,
        **kwargs: Any,
    ):
        self.highlighter = highlighter
        self.disable = disable
        super().__init__(self.highlighter(status), *args, console=console, **kwargs)

    def update(self, message: str) -> None:
        if not self.disable:
            super().update(self.highlighter(message))

    def start(self) -> None:
        if not self.disable:
            return super().start()

    def stop(self) -> None:
        if not self.disable:
            return super().stop()


class CustomConsole(Console):
    def __init__(self, theme_map: ThemeMap, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.theme_map = theme_map

    def _ensure_highlighter(
        self, text_theme_or_highlighter: TextTheme | SemanticHighlighter
    ) -> SemanticHighlighter:
        if isinstance(text_theme_or_highlighter, dict):
            return SemanticHighlighter(text_theme_or_highlighter)
        return text_theme_or_highlighter

    def done(self, message: str) -> None:
        self.print(self._ensure_highlighter(self.theme_map["done"])(message))

    def minor_done(self, message: str) -> None:
        self.print(self._ensure_highlighter(self.theme_map["minor_done"])(message))

    def warn(self, message: str) -> None:
        self.print(self._ensure_highlighter(self.theme_map["warn"])(message))

    def info(self, message: str) -> None:
        self.print(self._ensure_highlighter(self.theme_map["info"])(message))

    def progress(self) -> Progress:
        return Progress(
            SpinnerColumn(
                self.theme_map["progress"]["spinner"][0],
                style=self.theme_map["progress"]["spinner"][1],
                finished_text=self.theme_map["progress"]["spinner"][2],
            ),
            TextColumn(
                text_format="{task.description}",
                highlighter=self._ensure_highlighter(
                    self.theme_map["progress"]["desc"]
                ),
            ),
            BarColumn(
                style=self.theme_map["progress"]["bar"][0],
                complete_style=self.theme_map["progress"]["bar"][1],
                finished_style=self.theme_map["progress"]["bar"][2],
                pulse_style=self.theme_map["progress"]["bar"][3],
            ),
            TaskProgressColumn(
                text_format="{task.percentage:>3.0f}%",
                style=self.theme_map["progress"]["percent"],
            ),
            console=self,
            transient=True,
        )

    def track[T](
        self,
        iterable: Iterable[T],
        desc: str = "",
        total: int | None = None,
        disable: bool = False,
    ) -> Iterable[T]:
        if disable:
            yield from iterable
        else:
            with self.progress() as progress:
                yield from progress.track(iterable, description=desc, total=total)

    def status(self, message: str, disable: bool = False) -> CustomStatus:
        return CustomStatus(
            message,
            console=self,
            highlighter=self._ensure_highlighter(self.theme_map["status"]["desc"]),
            spinner=self.theme_map["status"]["spinner"][0],
            spinner_style=self.theme_map["status"]["spinner"][1],
            disable=disable,
        )

    def table(
        self,
        cols: Sequence[str] | None | pd.DataFrame = None,
        rows: Sequence[Sequence[Any]] | None = None,
        df: pd.DataFrame | None = None,
        *,
        justify: Sequence[Literal["left", "center", "right"]]
        | Literal["left", "center", "right"]
        | None = None,
        fmt_spec: Sequence[str] | str | None = None,
        title: str | None = None,
    ) -> None:
        # Infer Justify if fmt_spec is given
        if justify is None and fmt_spec is not None:
            if isinstance(fmt_spec, str):
                justify = determine_justify(fmt_spec)
            else:
                justify = [determine_justify(s) for s in fmt_spec]

        # Prepare justify iterator and fmt_spec iterator
        justify_iter: Iterable[Literal["left", "center", "right"]]
        if justify is None:
            justify_iter = repeat("left")  # ty: ignore
        elif isinstance(justify, str):
            justify_iter = repeat(justify)  # ty: ignore
        else:
            justify_iter = justify
        fmt_spec_iter: Iterable[str | None]
        if fmt_spec is None:
            fmt_spec_iter = repeat(None)
        elif isinstance(fmt_spec, str):
            fmt_spec_iter = repeat(fmt_spec)
        else:
            fmt_spec_iter = fmt_spec

        # Create table for rows and cols
        if rows is not None and cols is not None and df is None:
            assert not isinstance(cols, pd.DataFrame)
            table = Table(title=title)
            for col, j in zip(cols, justify_iter):
                j = cast(Literal["left", "center", "right"], j)
                table.add_column(col, justify=j)
            for row in rows:
                table.add_row(
                    *[
                        format_value(item, spec)
                        for item, spec in zip(row, fmt_spec_iter)
                    ]
                )
            self.print(table)
            return
        # Given a dataframe as positional argument
        if isinstance(cols, pd.DataFrame) and rows is None and df is None:
            df = cols
        # Create table for dataframe
        if df is not None:
            table = Table(title=title)
            for col, j in zip(df.columns, justify_iter):
                j = cast(Literal["left", "center", "right"], j)
                table.add_column(col, justify=j)
            for _, row in df.iterrows():
                table.add_row(
                    *[
                        format_value(item, spec)
                        for item, spec in zip(row, fmt_spec_iter)
                    ]
                )
            self.print(table)
        else:
            raise ValueError("Either rows and cols or df must be provided")

    @contextmanager
    def reroute_print(self, initial_message: str) -> Iterator[CustomStatus]:
        """Redirect all print calls to a Rich Status spinner for the duration of a context."""
        status = self.status(initial_message)
        original_print = builtins.print
        buffer = ""

        def patched_print(*args: Any, **kwargs: Any) -> None:
            nonlocal buffer
            sep = kwargs.pop("sep", " ")
            end = kwargs.pop("end", "\n")
            file = kwargs.pop("file", None)
            flush = kwargs.pop("flush", False)

            # If someone targets a specific file (other than stdout/stderr), leave it alone.
            if file not in (None, sys.stdout, sys.stderr):
                original_print(*args, sep=sep, end=end, file=file, flush=flush)
                return

            text = sep.join(map(str, args))
            buffer += text

            if end != "\n":
                buffer += end
                status.update(buffer)
            else:
                status.update(buffer)
                buffer = ""

            if flush:
                self.file.flush()

        with status:
            builtins.print = patched_print  # type: ignore
            try:
                yield status
            finally:
                builtins.print = original_print


THEME_MAP: ThemeMap = {
    "default": {
        "default": None,
        "number": "italic",
        "path": "underline",
        "first": None,
        "prefix": None,
        "prefix_style": None,
    },
    "done": {
        "default": None,
        "number": "italic",
        "path": "underline",
        "first": "bold",
        "prefix": None,
        "prefix_style": None,
    },
    "minor_done": {
        "default": "dim",
        "number": "italic",
        "path": "underline",
        "first": "bold",
        "prefix": None,
        "prefix_style": None,
    },
    "warn": {
        "default": None,
        "number": "italic",
        "path": "underline",
        "first": None,
        "prefix": "[WARN] ",
        "prefix_style": "bold yellow",
    },
    "info": {
        "default": None,
        "number": "italic",
        "path": "underline",
        "first": None,
        "prefix": "[INFO] ",
        "prefix_style": "bold cyan",
    },
    "progress": {
        "desc": {
            "default": "dim",
            "number": "italic",
            "path": "underline",
            "first": None,
            "prefix": None,
            "prefix_style": None,
        },
        "spinner": ("dots12", "dim", "  "),
        "bar": ("dim", "bold", "dim", "bold"),
        "percent": "white dim",
    },
    "status": {
        "desc": {
            "default": "dim",
            "number": "italic",
            "path": "underline",
            "first": None,
            "prefix": None,
            "prefix_style": None,
        },
        "spinner": ("dots12", "dim"),
    },
}

console = CustomConsole(THEME_MAP)


@overload
def with_status[**P, R](
    func: None = None,
    transient: bool = True,
    finished_text: str | None = None,
    include_args: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
@overload
def with_status[**P, R](
    func: Callable[P, R],
    transient: bool = True,
    finished_text: None = None,
    include_args: bool = True,
) -> Callable[P, R]: ...
def with_status[**P, R](
    func: Callable[P, R] | None = None,
    transient: bool = True,
    finished_text: str | None = None,
    include_args: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]] | Callable[P, R]:
    """Decorator factory that wraps function execution in a console status.

    Args:
        transient: If True, no message is shown after completion. If False, shows a completion message.
        finished_text: Custom message to show on completion. If None and transient is False,
                      defaults to "Completed function {func.__name__}".
        include_args: If True, includes function arguments in the status message.

    Shows "Running {func_name}({params})" while the function executes.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if kwargs.get("verbose", True):
                func_name = getattr(func, "__name__", "function")

                if include_args:
                    # Format parameters
                    params_str = _format_params(args, kwargs)
                    function_text = f"{func_name}({params_str})"
                else:
                    function_text = f"function {func_name}"

                start_time = time.perf_counter()
                with console.status(f"Running {function_text}"):
                    result = func(*args, **kwargs)
                elapsed_time = time.perf_counter() - start_time

                if not transient:
                    if finished_text is not None:
                        console.minor_done(finished_text)
                    else:
                        time_str = _format_elapsed_time(elapsed_time)
                        console.minor_done(f"Completed {function_text} in {time_str}")

                return result
            else:
                return func(*args, **kwargs)

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


def _format_elapsed_time(seconds: float) -> str:
    """Format elapsed time in appropriate units (ms, sec, min, or h)."""
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} sec"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} min"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} h"


def _format_params(args: tuple, kwargs: dict) -> str:
    """Format function parameters for display."""
    parts = []

    # Add positional args (no names)
    parts.extend(_format_value(arg) for arg in args)

    # Add keyword args (with names)
    parts.extend(f"{k}={_format_value(v)}" for k, v in kwargs.items())

    return ", ".join(parts)


def _format_value(value, max_length: int = 50) -> str:
    """Format a single value for display."""
    if isinstance(value, str):
        if len(value) > max_length:
            return f"'{value[: max_length - 3]}...'"
        return f"'{value}'"
    elif isinstance(value, (list, tuple, set)):
        if len(value) > 3:
            return f"{type(value).__name__}({len(value)} items)"
        repr_str = repr(value)
    elif isinstance(value, dict):
        if len(value) > 3:
            return f"dict({len(value)} items)"
        repr_str = repr(value)
    else:
        repr_str = repr(value)

    if len(repr_str) > max_length:
        return f"{type(value).__name__}(...)"
    return repr_str
