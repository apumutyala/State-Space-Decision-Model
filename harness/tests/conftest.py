import functools

import pytest

from harness import config, slicing


def suite_available():
    return (config.SUITE_DIR / "selected-rows.jsonl.gz").exists() and (config.SUITE_DIR / "added-rows.jsonl.gz").exists()


needs_suite = pytest.mark.skipif(not suite_available(), reason="local 0.2.1 suite not built (see harness/INDEX_NOTES.md)")


@functools.lru_cache(maxsize=1)
def _dev_rows():
    from harness.cli import load_slice

    return load_slice("dev")


@pytest.fixture(scope="session")
def dev_rows():
    if not suite_available():
        pytest.skip("local suite not built")
    return _dev_rows()


@pytest.fixture(scope="session")
def all_index_rows():
    if not suite_available():
        pytest.skip("local suite not built")
    from harness.suite import index_rows, load_suite

    return list(index_rows(load_suite()))
