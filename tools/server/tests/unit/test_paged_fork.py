import os
import pytest

from utils import ServerPreset

server = ServerPreset.tinyllama2()


# Long enough to fill at least one 16-token paged block so a named /fork
# child inherits by reference. A shorter prompt can still fork (name holds)
# but cache_n stays 0 -- that is honest, not the hole this test covers.
PREFIX = (
    "Once upon a time there was a little girl who lived in a village near the forest "
    "and she loved to walk among the trees and listen to the birds every morning"
)


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.extra_args = ["--kv-paged", "--kv-block-size", "16"]
    server.n_ctx = 256
    server.n_predict = 8
    # stories15M if the reviewer machine already has it; else tinyllama2/260K
    stories = os.environ.get("DS4P_STORIES15M")
    if stories and os.path.isfile(stories):
        server.model_file = stories
        server.model_hf_repo = None
        server.model_hf_file = None
        server.offline = False
    yield
    if server.process:
        server.stop()


def test_named_fork_child_reports_cache_n():
    """A named /fork child must report timings.cache_n != 0.

    The scheduler inherits whole physical blocks (logs: 'tokens inherited
    by reference'). HTTP used to leave cache_n=0, so the fork looked fake.
    """
    global server
    server.start()

    parent = server.make_request("POST", "/completion", data={
        "prompt": PREFIX,
        "n_predict": 4,
        "session_id": "master",
        "temperature": 0.0,
    })
    assert parent.status_code == 200, parent.body
    assert "timings" in parent.body

    child = server.make_request("POST", "/fork", data={
        "prompt": PREFIX + " and then she met a friend",
        "n_predict": 4,
        "parent_session_id": "master",
        "temperature": 0.0,
    })
    assert child.status_code == 200, child.body
    timings = child.body["timings"]
    cache_n = timings["cache_n"]
    assert cache_n > 0, (
        f"named /fork child reported cache_n=0; inherited prefix is missing "
        f"from HTTP JSON. timings={timings}"
    )
    assert cache_n % 16 == 0, (
        f"cache_n={cache_n} is not a whole-block multiple; fork_blocks "
        f"inherits physical blocks only"
    )
