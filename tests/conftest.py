import pytest

from mxagent.util import runlock


@pytest.fixture(autouse=True)
def _private_run_lock(tmp_path_factory, monkeypatch):
    """Ogni test ha il suo lock delle run (mxagent.util.runlock): i test che chiamano
    ``cli.main(["run"|"verify"|"eval", ...])`` non toccano il lock della macchina e non
    si bloccano se intanto gira una campagna."""
    monkeypatch.setenv(runlock.ENV_FILE, str(tmp_path_factory.mktemp("lock") / "run.lock"))
    monkeypatch.delenv(runlock.ENV_HELD, raising=False)
    yield
    runlock.release()
