"""The two remote texts of the paired coding evaluation, read as syntax trees.

These tests parse the texts the product builds and compare what the parent sends with them. They never run a
text: wrapper, answer and value code run only in remote sessions. What the remote service does with the texts is
measured outside this suite. The last two tests hold the rule the redirect rests on: the parent takes a reply
only when it is the whole standard output.
"""

from __future__ import annotations

import ast
import hashlib
import json

import pytest
from test_paired_coding_runner import Native, Oracle, population, run

from amlas.evaluation import coding_pair_runner as runner
from amlas.evaluation.coding_oracle import encode

ANSWER = "def fixture_entry(x):\n    return x\n\n\nFIXTURE_MARK = 1\n"
ENTRY = "fixture_entry"


def _load_body():
    return ast.parse(runner.bootstrap_code(ANSWER, ENTRY)).body


def _value_body(ordinal=3):
    wire = runner._canonical(encode(((1, 2), {"flag": True}))).decode()
    return ast.parse(runner.value_code(wire, ordinal)).body


def _is_stream_assignment(node, source):
    """``_amlas_sys.stdout = <source>`` and nothing else."""
    return isinstance(node, ast.Assign) and ast.unparse(node) == "_amlas_sys.stdout = " + source


def _is_reply(node):
    """A statement that prints one ``json.dumps`` call."""
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "print"
        and len(node.value.args) == 1
        and not node.value.keywords
        and isinstance(node.value.args[0], ast.Call)
        and ast.unparse(node.value.args[0].func) == "json.dumps"
    )


def _only(body, predicate):
    hits = [index for index, node in enumerate(body) if predicate(node)]
    assert len(hits) == 1, hits
    return hits[0]


def _called(node):
    return {
        call.func.id
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }


def test_load_text_redirects_before_the_answer_and_restores_before_the_reply():
    body = _load_body()
    save = _only(body, lambda node: ast.unparse(node) == "_amlas_out = _amlas_sys.stdout")
    null = _only(body, lambda node: ast.unparse(node) == "_amlas_null = open(_amlas_os.devnull, 'w')")
    redirect = _only(body, lambda node: _is_stream_assignment(node, "_amlas_null"))
    answer_first = _only(body, lambda node: isinstance(node, ast.FunctionDef) and node.name == ENTRY)
    answer_last = _only(body, lambda node: ast.unparse(node) == "FIXTURE_MARK = 1")
    entry = _only(body, lambda node: ast.unparse(node) == "_fn = " + ENTRY)
    restore = _only(body, lambda node: _is_stream_assignment(node, "_amlas_out"))
    reply = _only(body, _is_reply)
    assert save < null < redirect < answer_first < answer_last < entry < restore
    # The reply follows the restore directly and is the last statement.
    assert reply == restore + 1 == len(body) - 1
    assert ast.unparse(body[reply]) == "print(json.dumps({'ready': True}))"
    # The wrapper prints nothing else, and it never writes to a stream itself.
    assert [index for index, node in enumerate(body) if "print" in _called(node)] == [reply]


def test_load_text_keeps_the_answer_source_byte_for_byte():
    text = runner.bootstrap_code(ANSWER, ENTRY)
    head, redirected, rest = text.partition("_amlas_sys.stdout=_amlas_null\n")
    assert redirected and rest.startswith(ANSWER + "\n" + runner._GUEST_CODEC + "\n_fn=" + ENTRY + "\n")
    assert head.startswith("import json,math\n") and ANSWER not in head
    assert text.count(ANSWER) == 1


def test_value_text_runs_the_function_and_the_encoding_inside_the_redirect_and_restores_in_a_finally():
    body = _value_body(3)
    assert [type(node).__name__ for node in body] == ["Assign", "Assign", "Assign", "Try", "Expr"]
    assert ast.unparse(body[0]).startswith("_a, _k = _wire_decode(json.loads(")
    assert ast.unparse(body[1]) == "_amlas_out = _amlas_sys.stdout"
    assert _is_stream_assignment(body[2], "_amlas_null")
    guarded = body[3]
    assert not guarded.handlers and not guarded.orelse
    assert [ast.unparse(node) for node in guarded.body] == ["_amlas_v = _wire_encode(_fn(*_a, **_k))"]
    assert len(guarded.finalbody) == 1 and _is_stream_assignment(guarded.finalbody[0], "_amlas_out")
    assert _is_reply(body[4])
    assert ast.unparse(body[4]) == (
        "print(json.dumps({'ordinal': 3, 'value': _amlas_v}, sort_keys=True, allow_nan=False))"
    )
    # The answer's function and the encoding of its value are called only inside the guarded block.
    for node in (body[0], body[1], body[2], body[4]):
        assert not {"_fn", "_wire_encode"} & _called(node)


def test_value_text_carries_the_ordinal_and_the_arguments_it_was_given():
    wire = runner._canonical(encode(((1, 2), {"flag": True}))).decode()
    for ordinal in (1, 512):
        body = ast.parse(runner.value_code(wire, ordinal)).body
        assert ast.literal_eval(body[0].value.args[0].args[0]) == wire
        assert ast.unparse(body[4]).startswith(f"print(json.dumps({{'ordinal': {ordinal}, ")


def test_wrapper_texts_bind_no_module_beyond_json_math_and_two_private_names():
    tree = ast.parse(runner.bootstrap_code("", ENTRY))
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import | ast.ImportFrom)]
    assert all(isinstance(node, ast.Import) for node in imports)
    bound = {(alias.name, alias.asname or alias.name) for node in imports for alias in node.names}
    assert bound == {("json", "json"), ("math", "math"), ("os", "_amlas_os"), ("sys", "_amlas_sys")}
    # The wrapper reaches the two modules only through its private names: an answer that uses `os` or `sys`
    # without importing it finds no such name, as before.
    assert not {"os", "sys"} & {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    value = ast.parse(runner.value_code("[]", 1))
    assert not [node for node in ast.walk(value) if isinstance(node, ast.Import | ast.ImportFrom)]
    assert not {"os", "sys"} & {node.id for node in ast.walk(value) if isinstance(node, ast.Name)}
    assert runner.bootstrap_code(ANSWER, ENTRY).startswith(
        "import json,math\nimport os as _amlas_os,sys as _amlas_sys\n"
    )


def test_the_parent_sends_the_built_texts(tmp_path):
    result, native = run(tmp_path / "run")
    assert sum(row.outcome == "passed" for row in result["receipts"]) == 2
    cases, candidates = population()
    wire = runner._canonical(encode((((1, 2), {3: "v"}), {"flag": True}))).decode()
    for arm, session in (("base", "session-1"), ("tuned", "session-2")):
        sent = [request["arguments"]["code"] for request in native.invokes if request["sessionId"] == session]
        assert sent == [
            runner.bootstrap_code(candidates[arm][0]["source"], cases[0]["prompt"]["entry_point"]),
            runner.value_code(wire, 1),
        ]


def test_no_call_follows_a_failed_load_in_its_session(tmp_path):
    """A load that fails leaves the remote stream redirected, so the case must end without another call."""

    class FailedLoad(Native):
        def invoke_code_interpreter(self, **request):
            response = super().invoke_code_interpreter(**request)
            response["stream"] = iter(
                [
                    {
                        "result": {
                            "isError": True,
                            "content": [],
                            "structuredContent": {
                                "exitCode": 1,
                                "stdout": "",
                                "stderr": "neutral load failure",
                            },
                        }
                    }
                ]
            )
            return response

    root = tmp_path / "run"
    oracle = Oracle()
    result, native = run(root, oracle=oracle, native=FailedLoad(root))
    assert len(result["receipts"]) == 328
    assert [row.outcome for row in result["receipts"] if row.outcome != "incomplete"] == [
        "runtime_error",
        "runtime_error",
    ]
    # One call per session, the load, then the session is stopped and read back.
    assert [request["sessionId"] for request in native.invokes] == ["session-1", "session-2"]
    assert len(native.starts) == len(native.stops) == len(native.gets) == 2
    assert oracle.calls == []
    assert result["cleanup_complete"] is True


def _result_event(stdout):
    return {
        "result": {
            "isError": False,
            "content": [{"type": "text", "text": stdout}],
            "structuredContent": {"exitCode": 0, "stdout": stdout, "stderr": ""},
        }
    }


def test_the_reply_parser_takes_a_reply_only_when_it_is_the_whole_standard_output():
    reply = json.dumps({"ready": True})
    value, digest = runner._stdout((_result_event(reply),))
    assert value == {"ready": True}
    assert digest == hashlib.sha256(reply.encode()).hexdigest()
    # With the redirect in place, other output in the reply went around it. Such a reply is not taken.
    for stdout in (
        "side output\n" + reply,
        reply + "\n\nside output",
        reply + "\nside output",
        reply + reply,
        reply + "\n" + reply,
        "side output",
        "",
    ):
        with pytest.raises(ValueError):
            runner._stdout((_result_event(stdout),))


def test_output_beside_a_reply_ends_the_case_without_a_verdict_and_without_another_call(tmp_path):
    """Other output before or after a load reply or a value reply: the parent refuses the reply."""

    class OutputBesideTheReply(Native):
        def __init__(self, root, call, place):
            super().__init__(root)
            self.call, self.place = call, place

        def invoke_code_interpreter(self, **request):
            response = super().invoke_code_interpreter(**request)
            if sum(item["sessionId"] == request["sessionId"] for item in self.invokes) - 1 == self.call:
                reply = next(response["stream"])["result"]["structuredContent"]["stdout"]
                stdout = "side output\n" + reply if self.place == "before" else reply + "\n\nside output"
                response["stream"] = iter([_result_event(stdout)])
            return response

    for call, place, calls_sent, oracle_runs in (
        (0, "before", 2, 0),
        (0, "after", 2, 0),
        (1, "before", 4, 2),
        (1, "after", 4, 2),
    ):
        root = tmp_path / f"run-{call}-{place}"
        oracle = Oracle()
        result, native = run(root, oracle=oracle, native=OutputBesideTheReply(root, call, place))
        assert len(result["receipts"]) == 328
        assert [row.outcome for row in result["receipts"] if row.outcome != "incomplete"] == [
            "infrastructure_error",
            "infrastructure_error",
        ], (call, place)
        # The refused reply ends the case: no further call in that session, then one stop and one read back.
        assert len(native.invokes) == calls_sent and len(oracle.calls) == oracle_runs, (call, place)
        assert len(native.starts) == len(native.stops) == len(native.gets) == 2
        assert result["cleanup_complete"] is True
