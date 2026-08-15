import json
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
    """Named /fork child: cache_n is the inherited span, prompt_n is not.

    The scheduler inherits whole physical blocks. HTTP must report that
    as cache_n, and prompt_n must be only the tokens actually evaluated
    after that prefix (prompt_n + cache_n == tokens_evaluated). Parent
    cold stays cache_n=0. This is the HTTP proof; the C++ n_past test
    does not touch HTTP.
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
    parent_t = parent.body["timings"]
    parent_n = parent.body["tokens_evaluated"]
    assert parent_t["cache_n"] == 0, (
        f"parent cold must stay cache_n=0; timings={parent_t}"
    )
    assert parent_t["prompt_n"] + parent_t["cache_n"] == parent_n, (
        f"parent identity failed: prompt_n={parent_t['prompt_n']} "
        f"cache_n={parent_t['cache_n']} n_prompt={parent_n}"
    )

    child = server.make_request("POST", "/fork", data={
        "prompt": PREFIX + " and then she met a friend",
        "n_predict": 4,
        "parent_session_id": "master",
        "temperature": 0.0,
    })
    assert child.status_code == 200, child.body
    timings = child.body["timings"]
    cache_n = timings["cache_n"]
    prompt_n = timings["prompt_n"]
    n_prompt = child.body["tokens_evaluated"]
    out = os.environ.get("DS4P_FORK_CHILD_JSON")
    if out:
        with open(out, "w") as f:
            json.dump(child.body, f, indent=2)
            f.write("\n")
    assert cache_n > 0, (
        f"named /fork child reported cache_n=0; inherited prefix is missing "
        f"from HTTP JSON. timings={timings}"
    )
    assert cache_n % 16 == 0, (
        f"cache_n={cache_n} is not a whole-block multiple; fork_blocks "
        f"inherits physical blocks only"
    )
    # prompt_n is only tokens actually evaluated after the inherited prefix.
    # Billing the inherited span makes prompt_n + cache_n != n_prompt.
    assert prompt_n + cache_n == n_prompt, (
        f"child billed inherited tokens: prompt_n={prompt_n} cache_n={cache_n} "
        f"n_prompt={n_prompt} timings={timings}"
    )
    assert prompt_n < n_prompt, (
        f"child prompt_n={prompt_n} still equals full n_prompt={n_prompt}; "
        f"inherited span was billed"
    )
