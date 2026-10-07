# Receipt scripts (e.g. evasion_test.py) are run explicitly, not collected as tests.
collect_ignore_glob = ["receipts/*", "third_party/*"]

# ----------------------------------------------------------------------------- per-stack tests (LESSONS L-24)
# Jobs run on different pinned stacks (the Clef image pins transformers 5.10.2, the probe9b / rwkv7 / full8 images pin
# 5.17.0). A test whose behaviour depends on that pin carries @pytest.mark.requires_transformers("x.y.z") and is
# SKIPPED, never passed, under any other installed version. jobs/common/launch.py runs each job's tests under the job's
# own pin (job_params "test_env") and refuses the launch if a test that requires that pin was skipped.
REQUIRES_TRANSFORMERS_REASON = "requires_transformers {want} (installed {have})"


def installed_transformers():
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return None
    try:
        return version("transformers")
    except PackageNotFoundError:
        return None


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_transformers(version): run only under exactly this transformers "
                                       "version (the pinned stack of the job image the test checks); skipped otherwise")


def pytest_collection_modifyitems(config, items):
    import pytest

    have = installed_transformers()
    for item in items:
        m = item.get_closest_marker("requires_transformers")
        if m is None:
            continue
        want = m.args[0]
        if have != want:
            item.add_marker(pytest.mark.skip(reason=REQUIRES_TRANSFORMERS_REASON.format(want=want, have=have)))
