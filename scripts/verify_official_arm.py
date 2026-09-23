#!/usr/bin/env python
"""Verify that the official-replication arm sends ImpossibleBench's prompt VERBATIM.

`src/pp/runner_contract.py` claims -- in the docstrings of
`official_system_prompt()` and `official_system_prompt_for_runner()` -- that the
arm reproduces upstream's system prompt exactly, and that the ONLY difference
between what we send and what upstream sends is the appended
`OFFICIAL_RESPONSE_FORMAT` block. Until this script existed, nothing checked it.

WHAT THIS DOES
    1. Reconstructs the upstream prompt BY EXECUTING UPSTREAM'S OWN CODE. It does
       not re-type upstream's strings: it AST-parses the pinned checkout, lifts
       out `_get_default_template`, the `prompt_suffix` statements and the
       `test_info` statements, and runs them. Any upstream edit therefore moves
       the expected value automatically instead of silently agreeing with a stale
       copy of it.
    2. Reconstructs what WE send via `official_system_prompt_for_runner()`.
    3. Asserts ours == upstream + OFFICIAL_RESPONSE_FORMAT, over a matrix of
       argument combinations, and prints a unified diff of any other difference.
    4. Checks the STOP sentence against `demo.py`, the test-info disclosure
       against `swebench_agent_full.py:172-191`, and the tool set/timeouts.
    5. Checks the PER-TASK INTERPOLATION actually used by the study runners:
       upstream fills the TEST COMMAND line from `get_test_directives`, so the
       arguments our call sites pass must match that, not something else.

Exit code 0 iff every check passes; non-zero otherwise, with the divergence
quoted on both sides.

Self-check: `--tamper <kind>` mutates OUR prompt IN MEMORY (no file is written)
so you can confirm the script really fails when it should.

Static and offline. Runs no study, makes no model call, spends nothing.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import sys
import types
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable, Dict, List, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "studies"))

UPSTREAM_ROOT = REPO / "external" / "impossiblebench"
UPSTREAM_COMMIT = "061dc3dce6a96ab6cf02a855157263033dcfa3ba"
AGENT_FULL = UPSTREAM_ROOT / "src" / "impossiblebench" / "swebench_agent_full.py"
DEMO = UPSTREAM_ROOT / "demo.py"


# ===========================================================================
# failure accounting
# ===========================================================================
class Check:
    """One named fidelity check; holds its own failure text."""

    def __init__(self, cid: str, title: str) -> None:
        self.cid, self.title = cid, title
        self.failures: List[str] = []

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    @property
    def ok(self) -> bool:
        return not self.failures


CHECKS: List[Check] = []


def check(cid: str, title: str) -> Check:
    c = Check(cid, title)
    CHECKS.append(c)
    return c


def _vis(s: str) -> str:
    """Make whitespace-only differences legible in a diff."""
    return s.replace("\n", "\\n\n")


def unified(upstream: str, ours: str, *, a: str, b: str) -> str:
    """Unified diff, line-based, with trailing whitespace made visible."""
    def lines(t: str) -> List[str]:
        out = []
        for ln in t.split("\n"):
            marker = ""
            if ln != ln.rstrip():
                marker = "  <-- TRAILING WS " + repr(ln[len(ln.rstrip()):])
            out.append(ln + marker + "\n")
        return out

    return "".join(difflib.unified_diff(lines(upstream), lines(ours),
                                        fromfile=a, tofile=b, n=3))


# ===========================================================================
# upstream reconstruction -- by executing upstream's own statements
# ===========================================================================
def _parse(path: Path) -> Tuple[ast.Module, str]:
    src = path.read_text(encoding="utf-8")
    return ast.parse(src, filename=str(path)), src


def _find_funcdef(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == name:
            return node  # type: ignore[return-value]
    raise LookupError(f"no function {name!r} in the pinned upstream checkout")


def _exec_stmts(stmts: Sequence[ast.stmt], ns: Dict[str, Any],
                filename: str) -> Dict[str, Any]:
    mod = ast.Module(body=list(stmts), type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, filename, "exec"), ns)          # noqa: S102 - upstream src
    return ns


def _span(node: ast.AST) -> str:
    lo = getattr(node, "lineno", "?")
    hi = getattr(node, "end_lineno", lo)
    return f"{lo}" if lo == hi else f"{lo}-{hi}"


class Upstream:
    """Live handles on upstream's prompt-producing code."""

    def __init__(self) -> None:
        if not AGENT_FULL.exists():
            raise SystemExit(f"pinned upstream missing: {AGENT_FULL}")
        self.tree, self.src = _parse(AGENT_FULL)
        self.solver = _find_funcdef(self.tree, "multi_submission_solver")
        self.init = _find_funcdef(self.solver, "init")

        # --- _get_default_template (the whole template) --------------------
        tmpl_def = _find_funcdef(self.tree, "_get_default_template")
        self.template_span = _span(tmpl_def)
        ns = _exec_stmts([tmpl_def], {"dedent": dedent}, str(AGENT_FULL))
        self._template: Callable[[str, int, str], str] = ns["_get_default_template"]

        # --- prompt_suffix statements --------------------------------------
        body = self.solver.body
        lo = next(i for i, s in enumerate(body)
                  if isinstance(s, ast.Assign)
                  and getattr(s.targets[0], "id", None) == "extra_nudge")
        hi = next(i for i, s in enumerate(body)
                  if isinstance(s, ast.Assign)
                  and getattr(s.targets[0], "id", None) == "prompt_suffix"
                  and isinstance(s.value, ast.Call))
        self._suffix_stmts = body[lo:hi + 1]
        self.suffix_span = f"{body[lo].lineno}-{body[hi].end_lineno}"

        # --- test_info statements ------------------------------------------
        ibody = self.init.body
        ti = next(i for i, s in enumerate(ibody)
                  if isinstance(s, ast.Assign)
                  and getattr(s.targets[0], "id", None) == "test_info")
        nxt = ibody[ti + 1]
        if not isinstance(nxt, ast.If):
            raise SystemExit("upstream test_info shape changed: expected an `if` "
                             f"after `test_info = \"\"` at line {ibody[ti].lineno}")
        self._testinfo_stmts = [ibody[ti], nxt]
        self.testinfo_span = f"{ibody[ti].lineno}-{nxt.end_lineno}"

        # --- tools ----------------------------------------------------------
        self.tools = self._read_tools()

        # --- demo.py knobs ---------------------------------------------------
        self.demo_custom_prompt, self.demo_human_intervention, self.demo_lines = \
            self._read_demo()

    # -- reconstructions ---------------------------------------------------
    def prompt_suffix(self, custom_prompt: str, human_intervention: bool) -> str:
        ns = _exec_stmts(self._suffix_stmts,
                         {"custom_prompt": custom_prompt,
                          "human_intervention": human_intervention},
                         str(AGENT_FULL))
        return ns["prompt_suffix"]

    def test_info(self, test_command: str, test_files: Sequence[str]) -> str:
        ns = _exec_stmts(self._testinfo_stmts,
                         {"test_command": test_command,
                          "test_files": list(test_files)},
                         str(AGENT_FULL))
        return ns["test_info"]

    def system_prompt(self, *, test_command: str = "",
                      test_files: Sequence[str] = (),
                      max_attempts: int = 10,
                      human_intervention: bool = False,
                      custom_prompt: str | None = None) -> str:
        cp = self.demo_custom_prompt if custom_prompt is None else custom_prompt
        return self._template(self.test_info(test_command, test_files),
                              max_attempts,
                              self.prompt_suffix(cp, human_intervention))

    # -- structural reads ---------------------------------------------------
    def _read_tools(self) -> List[Tuple[str, Dict[str, str], int]]:
        """[(tool_name, {kwarg: literal-or-source}, lineno)] from basic_agent()."""
        ret = next(s for s in reversed(self.solver.body) if isinstance(s, ast.Return))
        call = ret.value
        assert isinstance(call, ast.Call)
        tools_kw = next(k for k in call.keywords if k.arg == "tools")
        assert isinstance(tools_kw.value, ast.List)
        out: List[Tuple[str, Dict[str, str], int]] = []
        for el in tools_kw.value.elts:
            assert isinstance(el, ast.Call)
            name = getattr(el.func, "id", getattr(el.func, "attr", "?"))
            kw: Dict[str, str] = {}
            for k in el.keywords:
                kw[str(k.arg)] = ast.unparse(k.value)
            out.append((str(name), kw, el.lineno))
        return out

    def _read_demo(self) -> Tuple[str, bool, Dict[str, int]]:
        tree, _ = _parse(DEMO)
        fn = _find_funcdef(tree, "replicate_swebench")
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and \
                    getattr(node.func, "id", None) == "impossible_swebench":
                kws = {k.arg: k for k in node.keywords}
                cp = kws["custom_prompt"]
                hi = kws["human_intervention"]
                return (ast.literal_eval(cp.value),
                        bool(ast.literal_eval(hi.value)),
                        {"custom_prompt": cp.value.lineno,
                         "human_intervention": hi.value.lineno})
        raise SystemExit("demo.py: no impossible_swebench(...) call found")


# ===========================================================================
# our side
# ===========================================================================
def load_ours(tamper: str | None) -> types.SimpleNamespace:
    import pp.runner_contract as rc

    response_format = rc.OFFICIAL_RESPONSE_FORMAT
    action_vocabulary = rc.SHARED_ACTION_VOCABULARY
    base = rc.official_system_prompt

    if tamper:
        # IN-MEMORY ONLY. Nothing is written to disk. This exists so the gate can
        # be shown to fail when it should, per VERIFY-YOUR-OWN-WORK.
        def mutate(text: str) -> str:
            if tamper == "reword-tip":
                return text.replace(
                    "5. Identify the root cause of the issue",
                    "5. Identify the root cause of the bug")
            if tamper == "drop-tool":
                return text.replace("- think: Log thoughts and reasoning\n", "")
            if tamper == "extra-sentence":
                return text.replace(
                    "Some tips:",
                    "Remember that the tests are authoritative.\n\nSome tips:")
            if tamper == "whitespace":
                return text.replace("TOOLS AVAILABLE:", "TOOLS AVAILABLE: ")
            if tamper == "stop-sentence":
                return text.replace("STOP.", "stop immediately.")
            if tamper == "reveal-more":
                return text.replace(
                    "Some tips:",
                    "TEST FILES TO EXAMINE:\n  - tests/test_x.py\n\nSome tips:")
            raise SystemExit(f"unknown --tamper kind {tamper!r}")

        def tampered_for_runner(**kw: Any) -> str:
            return mutate(base(**kw)) + response_format

        for_runner = tampered_for_runner
    else:
        for_runner = rc.official_system_prompt_for_runner

    return types.SimpleNamespace(
        rc=rc, base=base, for_runner=for_runner,
        response_format=response_format,
        action_vocabulary=action_vocabulary,
        custom_prompt=rc.OFFICIAL_CUSTOM_PROMPT,
        human_sentence=rc.OFFICIAL_HUMAN_INTERVENTION_SENTENCE,
        suffix=rc.official_prompt_suffix,
        test_info=rc.official_test_info,
    )


# ===========================================================================
# the argument matrix -- both the paper's configuration and the edge cases
# ===========================================================================
CASES: List[Dict[str, Any]] = [
    dict(name="paper config, django task (real per-task interpolation)",
         test_command="./tests/runtests.py --verbosity 2 --settings=test_sqlite "
                      "--parallel 1",
         test_files=["queries.test_qs_combinators"],
         max_attempts=10, human_intervention=False),
    dict(name="paper config, pytest task, two test files",
         test_command="pytest -rA",
         test_files=["astropy/units/tests/test_quantity.py",
                     "astropy/units/tests/test_quantity_ufuncs.py"],
         max_attempts=10, human_intervention=False),
    dict(name="no test info (hide_tests / missing version path)",
         test_command="", test_files=[],
         max_attempts=10, human_intervention=False),
    dict(name="test_command without files (upstream emits nothing)",
         test_command="pytest -rA", test_files=[],
         max_attempts=10, human_intervention=False),
    dict(name="human_intervention=True (the other upstream experiment)",
         test_command="pytest -rA", test_files=["tests/test_x.py"],
         max_attempts=10, human_intervention=True),
    dict(name="max_attempts=3",
         test_command="pytest -rA", test_files=["tests/test_x.py"],
         max_attempts=3, human_intervention=False),
]


# ===========================================================================
# checks
# ===========================================================================
def check_template(up: Upstream, ours: types.SimpleNamespace) -> None:
    c = check("A", "our runner prompt == upstream prompt + the appended blocks")
    # TWO appended blocks as of 2026-09-18:
    #   OFFICIAL_RESPONSE_FORMAT  -- the JSON envelope our parser needs
    #   SHARED_ACTION_VOCABULARY  -- the seven action tokens, held CONSTANT
    #                                across the grid
    # The vocabulary was added because upstream's template names no actions, so
    # an agent under it had no ESCALATE_TO_HUMAN token and could not escalate at
    # all -- making the study's primary escalation outcome inexpressible in that
    # arm. Neither block states a scope rule, ranks an action, or names an
    # out-of-scope action, so upstream's NORMATIVE content is still untouched,
    # and this check still proves the underlying template is verbatim.
    appended = ours.response_format + ours.action_vocabulary
    for case in CASES:
        kw = {k: v for k, v in case.items() if k != "name"}
        want = up.system_prompt(**kw) + appended
        got = ours.for_runner(**kw)
        if got == want:
            continue
        d = unified(want, got,
                    a=f"UPSTREAM {AGENT_FULL.relative_to(REPO)}:{up.template_span}"
                      " + OFFICIAL_RESPONSE_FORMAT + SHARED_ACTION_VOCABULARY",
                    b="OURS src/pp/runner_contract.py official_system_prompt_for_runner()")
        c.fail(f"case {case['name']!r}: our prompt is NOT upstream + the appended "
               f"block.\n{d}")

    # And separately: the appended block really is a pure suffix of what we send.
    kw = {k: v for k, v in CASES[0].items() if k != "name"}
    got = ours.for_runner(**kw)
    if not got.endswith(appended):
        c.fail("the appended blocks are not a suffix of what we send; the "
               "'only difference is an appended block' claim cannot hold.")
    stripped = got[: -len(appended)]
    if stripped != up.system_prompt(**kw):
        c.fail("after removing the appended block, the remainder still differs "
               "from upstream (see the diff above).")


def check_stop_sentence(up: Upstream, ours: types.SimpleNamespace) -> None:
    c = check("B", "STOP sentence matches demo.py verbatim")
    if ours.custom_prompt != up.demo_custom_prompt:
        c.fail(
            "OFFICIAL_CUSTOM_PROMPT != demo.py custom_prompt.\n"
            f"  upstream demo.py:{up.demo_lines['custom_prompt']}: "
            f"{up.demo_custom_prompt!r}\n"
            f"  ours     src/pp/runner_contract.py OFFICIAL_CUSTOM_PROMPT: "
            f"{ours.custom_prompt!r}")
    if up.demo_human_intervention is not False:
        c.fail(f"demo.py:{up.demo_lines['human_intervention']} now sets "
               f"human_intervention={up.demo_human_intervention}; our replication "
               "assumes False.")
    # the suffix assembly itself, both branches
    for hi in (False, True):
        want = up.prompt_suffix(up.demo_custom_prompt, hi)
        got = ours.suffix(human_intervention=hi)
        if want != got:
            c.fail(f"official_prompt_suffix(human_intervention={hi}) diverges.\n"
                   f"  upstream {AGENT_FULL.relative_to(REPO)}:{up.suffix_span}: "
                   f"{want!r}\n  ours: {got!r}")


def check_test_info(up: Upstream, ours: types.SimpleNamespace) -> None:
    c = check("C", "test-info disclosure parity (no more, no less than upstream)")
    probes = [
        ("pytest -rA", ["a/test_x.py"]),
        ("pytest -rA", ["a/test_x.py", "b/test_y.py"]),
        ("./tests/runtests.py --verbosity 2", ["queries.test_qs_combinators"]),
        ("", ["a/test_x.py"]),
        ("pytest -rA", []),
        ("", []),
    ]
    for cmd, files in probes:
        want = up.test_info(cmd, files)
        got = ours.test_info(cmd, files)
        if want != got:
            c.fail(
                f"official_test_info({cmd!r}, {files!r}) diverges.\n"
                f"  upstream {AGENT_FULL.relative_to(REPO)}:{up.testinfo_span}: "
                f"{want!r}\n  ours     src/pp/runner_contract.py "
                f"official_test_info(): {got!r}")

    # the three blocks upstream COMMENTED OUT must not be live on either side
    dead = ("TESTS THAT MUST PASS", "TESTS THAT MUST CONTINUE TO PASS",
            "TEST FILES TO EXAMINE")
    live_up = up.test_info("pytest -rA", ["a/test_x.py"])
    live_ours = ours.test_info("pytest -rA", ["a/test_x.py"])
    for phrase in dead:
        if phrase in live_up:
            c.fail(f"upstream now EMITS {phrase!r} (it was commented out at "
                   f"{AGENT_FULL.relative_to(REPO)}:178-191); our arm does not.")
        if phrase in live_ours:
            c.fail(f"we emit {phrase!r}, which upstream keeps commented out at "
                   f"{AGENT_FULL.relative_to(REPO)}:178-191 -- our arm would be "
                   "EASIER to cheat than official.")


def check_tools(up: Upstream, ours: types.SimpleNamespace) -> None:
    c = check("D", "4 tools, timeout=60, and the TOOLS AVAILABLE block")
    names = [t[0] for t in up.tools]
    if names != ["bash", "python", "text_editor", "think"]:
        c.fail(f"upstream tool list changed: {names}")
    for name, kw, ln in up.tools:
        if name == "think":
            if "timeout" in kw:
                c.fail(f"think() now takes timeout={kw['timeout']} at "
                       f"{AGENT_FULL.relative_to(REPO)}:{ln}")
            continue
        if kw.get("timeout") != "60":
            c.fail(f"{name}() timeout is {kw.get('timeout')!r}, not 60, at "
                   f"{AGENT_FULL.relative_to(REPO)}:{ln}")

    # the prompt's own TOOLS AVAILABLE block, compared section-to-section
    up_block = _section(up.system_prompt(**_kw(CASES[0])), "TOOLS AVAILABLE:")
    our_block = _section(ours.for_runner(**_kw(CASES[0])), "TOOLS AVAILABLE:")
    if up_block != our_block:
        c.fail("TOOLS AVAILABLE block differs.\n" +
               unified(up_block, our_block, a="upstream", b="ours"))
    for tool in ("bash", "python", "text_editor", "think"):
        if f"- {tool}:" not in our_block:
            c.fail(f"our TOOLS AVAILABLE block does not list {tool!r}")


def check_tips(up: Upstream, ours: types.SimpleNamespace) -> None:
    c = check("E", "the 8 numbered procedural tips match word-for-word")
    up_block = _section(up.system_prompt(**_kw(CASES[0])), "Some tips:")
    our_block = _section(ours.for_runner(**_kw(CASES[0])), "Some tips:")
    up_tips = [ln for ln in up_block.split("\n") if ln[:1].isdigit()]
    our_tips = [ln for ln in our_block.split("\n") if ln[:1].isdigit()]
    if len(up_tips) != 8:
        c.fail(f"upstream no longer has 8 tips (has {len(up_tips)})")
    if up_tips != our_tips:
        c.fail("numbered tips differ.\n" +
               unified("\n".join(up_tips), "\n".join(our_tips),
                       a="upstream", b="ours"))


def advisories(up: Upstream, ours: types.SimpleNamespace) -> List[str]:
    """Non-fatal observations ABOUT the appended block itself.

    Check A deliberately excuses `OFFICIAL_RESPONSE_FORMAT` from the diff -- that
    is the whole point of the adapter. But the adapter is still text the model
    reads, and `src/pp/runner_contract.py` justifies it with the claim that it
    "names no new capability (the tools are already listed in official's own
    TOOLS AVAILABLE block)". That claim is checkable, so check it and say so
    without failing the gate.
    """
    out: List[str] = []
    block = ours.response_format
    upstream_tools = [t[0] for t in up.tools]
    named = [t for t in upstream_tools if f'"{t}"' in block]
    missing = [t for t in upstream_tools if f'"{t}"' not in block]
    if missing:
        out.append(
            "OFFICIAL_RESPONSE_FORMAT enumerates the tools the model may name, "
            f"and omits {missing!r}, which upstream's TOOLS AVAILABLE block does "
            "advertise. The arm therefore advertises a tool in one paragraph and "
            "leaves it unreachable in the transport paragraph.")
    for extra in ("run_tests",):
        if f'"{extra}"' in block and extra not in upstream_tools:
            out.append(
                f"OFFICIAL_RESPONSE_FORMAT names {extra!r}, which is not one of "
                f"upstream's four tools ({upstream_tools}); upstream exposes test "
                "feedback through basic_agent's own submit mechanism instead "
                "(swebench_agent_full.py:228-250). The docstring claim that the "
                "block 'names no new capability' is not literally true.")
    if named:
        out.append(f"tools named in the block and also upstream: {named}")
    return out


def _kw(case: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in case.items() if k != "name"}


def _section(text: str, header: str) -> str:
    """The header line plus every following line up to the next blank line."""
    lines = text.split("\n")
    try:
        i = lines.index(header)
    except ValueError:
        return ""
    out = [lines[i]]
    for ln in lines[i + 1:]:
        if not ln.strip():
            break
        out.append(ln)
    return "\n".join(out)


def check_interpolation() -> None:
    """What the STUDY RUNNERS actually feed into the per-task TEST COMMAND line.

    Upstream fills `test_files` from `get_test_directives` (`:167`) and then
    prints `f"{test_command} {' '.join(test_files)}"` (`:175`). If our call sites
    feed something else, the sentence the model reads is not upstream's sentence
    even when the template around it is.
    """
    c = check("F", "per-task interpolation matches upstream's get_test_directives")
    try:
        from pp.task_environments import (FROZEN_SPEC_PATH, load_frozen_specs,
                                          load_swebench, load_raw_rows)
        import study_ab_runner as sar
    except Exception as e:                                     # noqa: BLE001
        c.fail(f"could not load the task tables to check interpolation: {e!r}")
        return

    try:
        specs = load_frozen_specs(FROZEN_SPEC_PATH)
        swb = load_swebench()
        raw = {v: load_raw_rows(v) for v in ("conflicting", "original")}
    except Exception as e:                                     # noqa: BLE001
        c.fail(f"task tables unavailable: {e!r}")
        return

    bad: List[str] = []
    for iid, spec in sorted(specs.items()):
        for variant in ("conflicting", "original"):
            row = raw[variant].get(iid)
            if not row or not row.get("test_patch"):
                continue
            # upstream: swebench_agent_full.py:167 and :170
            up_files = list(swb.get_test_directives(
                {"repo": spec.repo, "test_patch": row["test_patch"]}))
            up_cmd = swb.MAP_REPO_VERSION_TO_SPECS[spec.repo][spec.version]["test_cmd"]
            # ours: studies/study_ab_runner.py:1358-1378
            our_cmd = sar._official_test_command(iid)
            our_files = sar._official_test_files(iid, variant)
            if (up_cmd, up_files) != (our_cmd, our_files):
                bad.append(
                    f"  {iid} [{variant}]\n"
                    f"    upstream line: {up_cmd} {' '.join(up_files)}\n"
                    f"    ours     line: {our_cmd} {' '.join(our_files)}")
    if bad:
        c.fail(
            "the per-task TEST COMMAND sentence differs from upstream's for "
            f"{len(bad)} task/variant pairs. Upstream joins the command with "
            "get_test_directives() output (external/impossiblebench/src/"
            "impossiblebench/swebench_agent_full.py:167,175); our call sites "
            "(studies/study_ab_runner.py:1368-1378 `_official_test_files`, and "
            "scripts/run_study_b_swarm.py:59-63) pass `spec.test_files`, which "
            "src/pp/task_environments.py:317 fills from get_modified_files() "
            "rather than get_test_directives() (that is at :316):\n"
            + "\n".join(bad))


# ===========================================================================
# main
# ===========================================================================
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tamper", default=None,
                    choices=["reword-tip", "drop-tool", "extra-sentence",
                             "whitespace", "stop-sentence", "reveal-more"],
                    help="mutate OUR prompt in memory (no file is touched) to "
                         "prove this gate fails when fidelity is broken")
    ap.add_argument("--skip-interpolation", action="store_true",
                    help="only check the template, not the per-task arguments "
                         "the study runners feed it")
    ap.add_argument("--print-prompt", action="store_true",
                    help="print the reconstructed upstream prompt and exit")
    args = ap.parse_args(argv)

    up = Upstream()
    if args.print_prompt:
        sys.stdout.write(up.system_prompt(**_kw(CASES[0])))
        return 0

    ours = load_ours(args.tamper)

    print(f"pinned upstream : {UPSTREAM_ROOT.relative_to(REPO)} @ {UPSTREAM_COMMIT}")
    print(f"template        : {AGENT_FULL.relative_to(REPO)}:{up.template_span}"
          "  (_get_default_template)")
    print(f"prompt_suffix   : {AGENT_FULL.relative_to(REPO)}:{up.suffix_span}")
    print(f"test_info       : {AGENT_FULL.relative_to(REPO)}:{up.testinfo_span}")
    print(f"custom_prompt   : demo.py:{up.demo_lines['custom_prompt']}")
    if args.tamper:
        print(f"!! TAMPER MODE  : our prompt mutated in memory ({args.tamper}); "
              "this run is EXPECTED to fail")
    print()

    check_template(up, ours)
    check_stop_sentence(up, ours)
    check_test_info(up, ours)
    check_tools(up, ours)
    check_tips(up, ours)
    if not args.skip_interpolation:
        check_interpolation()

    failed = [c for c in CHECKS if not c.ok]
    for c in CHECKS:
        print(f"[{'PASS' if c.ok else 'FAIL'}] {c.cid}. {c.title}")
    print()
    adv = advisories(up, ours)
    if adv:
        print("--- ADVISORY (about the appended block; not part of the gate) ---")
        for a in adv:
            print(f"  * {a}")
        print()
    for c in failed:
        print(f"--- DIVERGENCE ({c.cid}. {c.title}) " + "-" * 20)
        for f in c.failures:
            print(f)
            print()

    if failed:
        print("VERDICT: the official-replication arm does NOT send ImpossibleBench's "
              "prompt verbatim. Failing checks: "
              + ", ".join(c.cid for c in failed))
        return 1
    print("VERDICT: the official-replication arm sends upstream's prompt verbatim; "
          "the only difference is the appended OFFICIAL_RESPONSE_FORMAT block.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
