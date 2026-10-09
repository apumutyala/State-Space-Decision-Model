import pytest

# Receipt scripts (e.g. evasion_test.py) are run explicitly, not collected as tests.
collect_ignore_glob = ["receipts/*", "third_party/*"]

# ----------------------------------------------------------------------------- git env isolation
# An exported GIT_DIR/GIT_WORK_TREE makes a test's `git init <tmp>` act on the REAL repository: on 2026-10-08 that wrote
# core.worktree into the shared .git/config and broke every Windows git command. Scrub them for the whole session.
SCRUBBED_GIT_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR", "GIT_CEILING_DIRECTORIES")


@pytest.fixture(scope="session", autouse=True)
def _scrub_git_env():
    mp = pytest.MonkeyPatch()
    for k in SCRUBBED_GIT_ENV:
        mp.delenv(k, raising=False)
    try:
        yield
    finally:
        mp.undo()


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


# ----------------------------------------------------------------------------- doc-overlap guard (train.doc_overlap_guard)
# Every in-process caller of train_set.draw / run_cell.load_cell_data / heldout_gate / clef_teacher.select now runs the
# doc-overlap gate, which needs the eval hash index of the real Index suite (absent on job boxes' test envs). Unit tests
# get an EMPTY synthetic index (no eval grams: the gate passes) unless they install their own
# (tests/test_doc_overlap_guard.py does, with copied / paraphrased / allowlisted synthetic eval documents).
@pytest.fixture(autouse=True)
def _doc_overlap_test_index():
    try:
        from train import doc_overlap_guard as DG
    except Exception:  # pragma: no cover  (numpy missing: the gate cannot be imported by the code under test either)
        yield
        return
    empty = DG.build_index(None, {"entries": []}, rows_factory=lambda: iter([]), n_parts=1)
    empty.meta["index_npz_sha256"] = "empty-test-index"
    DG.set_index_for_tests(empty)
    try:
        yield
    finally:
        DG.set_index_for_tests(None)
