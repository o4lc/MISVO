"""MBPP+ scorer: verifiable code-generation scoring via test-case execution."""

import json
import logging
import os
import re
import resource
import subprocess
import sys

logger = logging.getLogger(__name__)


def _limit_memory():
    """Preexec function to cap child process memory at 256 MB (test_list mode)."""
    limit = 256 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def _limit_memory_plus():
    """No-op preexec for plus mode — numpy/OpenBLAS require large virtual
    address space that conflicts with RLIMIT_AS. Plus mode is evaluation-only
    (not called on model-generated untrusted code at training time), and the
    subprocess timeout still bounds runtime.
    """
    pass


class MBPPPlusScorer:
    """Score code-generation responses by executing MBPP+ test cases.

    Interface mirrors IFBenchScorer so all run scripts can use it via the
    same duck-typed protocol.
    """

    def __init__(self, data_path=None, timeout=5, timeout_plus=15):
        self.timeout = timeout
        self.timeout_plus = timeout_plus
        if data_path is None:
            data_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "mbpp_plus", "mbppplus.jsonl"
            )
        self.data_path = data_path
        self._entries = []
        self._prompt_index = {}     # raw prompt text -> entry
        self._stripped_index = {}   # stripped formatted prompt -> entry

    def load(self):
        with open(self.data_path, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                self._entries.append(entry)
                self._prompt_index[entry["prompt"]] = entry
        # Build stripped index for lookup from formatted prompts
        for entry in self._entries:
            stripped = self._strip_formatting(entry["prompt"])
            if stripped:
                self._stripped_index[stripped] = entry
        logger.info(f"MBPPPlusScorer: loaded {len(self._entries)} problems")

    def unload(self):
        pass

    def get_prompts(self):
        """Return raw prompt strings (for dataset loading compatibility)."""
        return [e["prompt"] for e in self._entries]

    # ------------------------------------------------------------------
    # Prompt lookup
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_formatting(text):
        """Strip the EvalPlus-style instruction wrapper to recover raw prompt."""
        # Extract the docstring-wrapped task prompt from the instruction
        # format: ```\n"""\n{desc}\n{assertion}\n"""\n```
        m = re.search(
            r'in a markdown code block:\s*```\s*\n"""\s*\n(.*?)\n[^\n]*\n"""\s*\n\s*```',
            text, re.DOTALL,
        )
        if m:
            return m.group(1).strip()
        # Fallback: any markdown code block
        m = re.search(
            r"in a markdown code block:\s*```\s*\n(.*?)\n\s*```",
            text, re.DOTALL,
        )
        if m:
            return m.group(1).strip()
        # Try simpler User:/Assistant: wrapper
        m = re.search(r"User:\s*(.*?)(?:\nAssistant:)", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return text.strip()

    def _lookup_entry(self, prompt):
        """Map a (possibly formatted) prompt to its MBPP+ entry."""
        if prompt in self._prompt_index:
            return self._prompt_index[prompt]
        stripped = self._strip_formatting(prompt)
        if stripped in self._stripped_index:
            return self._stripped_index[stripped]
        # Fuzzy: check if any raw prompt is a substring of the formatted prompt
        for raw, entry in self._prompt_index.items():
            if raw in prompt:
                return entry
        return None

    # ------------------------------------------------------------------
    # Code extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_code(response):
        """Extract Python code from an LLM response.

        The assistant prefix already opens a ```python block, so the
        response is typically raw code terminated by ```.
        """
        if not response or not response.strip():
            return ""
        text = response

        # Case 1: model repeated a ```python ... ``` block
        m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()

        # Case 2: response ends with closing ``` (from pre-filled prefix)
        if "```" in text:
            return text.split("```")[0].strip()

        # Case 3: raw code
        return text.strip()

    # ------------------------------------------------------------------
    # Test execution
    # ------------------------------------------------------------------

    def _execute_tests(self, code, entry):
        """Run all test_list assertions in a single subprocess.

        Returns list of bools (one per assertion).
        """
        test_list = entry.get("test_list", [])
        if not test_list:
            return []

        # Build imports
        imports = "\n".join(entry.get("test_imports", []))

        # Build script: code, imports, then test each assertion
        parts = [code]
        if imports:
            parts.append(imports)
        parts.append("_results = []")
        for assertion in test_list:
            # Each assertion wrapped in try/except
            parts.append(
                f"try:\n    {assertion}\n    _results.append(1)\nexcept:\n    _results.append(0)"
            )
        parts.append("print(','.join(map(str, _results)))")
        script = "\n".join(parts)

        try:
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                timeout=self.timeout,
                text=True,
                preexec_fn=_limit_memory,
                env={"PATH": os.environ.get("PATH", "")},
            )
            if result.returncode != 0:
                return [False] * len(test_list)
            stdout = result.stdout.strip()
            if not stdout:
                return [False] * len(test_list)
            return [p == "1" for p in stdout.split(",")]
        except subprocess.TimeoutExpired:
            return [False] * len(test_list)
        except Exception:
            return [False] * len(test_list)

    # ------------------------------------------------------------------
    # Scoring methods
    # ------------------------------------------------------------------

    def score(self, prompt, response):
        """Binary score: 1.0 if all tests pass, 0.0 otherwise."""
        entry = self._lookup_entry(prompt)
        if entry is None:
            logger.warning(f"MBPPPlusScorer: no entry found for prompt (first 80 chars): {prompt[:80]}")
            return 0.0
        code = self._extract_code(response)
        if not code:
            return 0.0
        results = self._execute_tests(code, entry)
        if not results:
            return 0.0
        return 1.0 if all(results) else 0.0

    def score_float(self, prompt, response):
        """Continuous score in [-1, 1]: 2 * (passed/total) - 1."""
        entry = self._lookup_entry(prompt)
        if entry is None:
            logger.warning(f"MBPPPlusScorer: no entry found for prompt (first 80 chars): {prompt[:80]}")
            return -1.0
        code = self._extract_code(response)
        if not code:
            return -1.0
        results = self._execute_tests(code, entry)
        if not results:
            return -1.0
        passed = sum(results)
        return 2.0 * (passed / len(results)) - 1.0

    def score_both(self, prompt, response):
        """Return (binary_score, float_score) from a single execution."""
        entry = self._lookup_entry(prompt)
        if entry is None:
            return 0.0, -1.0
        code = self._extract_code(response)
        if not code:
            return 0.0, -1.0
        results = self._execute_tests(code, entry)
        if not results:
            return 0.0, -1.0
        passed = sum(results)
        total = len(results)
        binary = 1.0 if passed == total else 0.0
        continuous = 2.0 * (passed / total) - 1.0
        return binary, continuous

    def score_responses(self, prompts, responses, batch_size=None):
        """Batch binary scoring."""
        return [self.score(p, r) for p, r in zip(prompts, responses)]

    def score_responses_float(self, prompts, responses, batch_size=None):
        """Batch float scoring."""
        return [self.score_float(p, r) for p, r in zip(prompts, responses)]

    def score_responses_both(self, prompts, responses, batch_size=None):
        """Batch scoring returning (binary_list, float_list)."""
        binary_scores = []
        float_scores = []
        for p, r in zip(prompts, responses):
            b, f = self.score_both(p, r)
            binary_scores.append(b)
            float_scores.append(f)
        return binary_scores, float_scores

    # ------------------------------------------------------------------
    # Plus-mode scoring: full MBPP+ augmented test suite (100+ tests).
    # Used for evaluation only — training paths keep using test_list.
    # ------------------------------------------------------------------

    def _execute_full_test(self, code, entry):
        """Run the full augmented MBPP+ test as a single subprocess.

        Returns (passed: bool, n_passed: int, n_total: int) by monkey-patching
        the `assertion` function defined in entry["test"] to count pass/fail
        instead of raising on first failure.
        """
        test_code = entry.get("test", "")
        if not test_code:
            return False, 0, 0

        # Count assertions in the test (each `assertion(...)` call = one test).
        # This is a heuristic; we override `assertion` at runtime to get the
        # true count via per-call tracking.
        counter_wrapper = (
            "\n\n# --- plus-mode instrumentation ---\n"
            "__orig_assertion = assertion\n"
            "__plus_passed = [0]\n"
            "__plus_total = [0]\n"
            "def assertion(out, exp, atol):\n"
            "    __plus_total[0] += 1\n"
            "    try:\n"
            "        __orig_assertion(out, exp, atol)\n"
            "        __plus_passed[0] += 1\n"
            "    except BaseException:\n"
            "        pass\n"
        )

        # The original `test` field ends with a for-loop that calls
        # `assertion(func(*inp), exp, atol)`. We need to (1) define helpers and
        # build inputs/results, (2) monkey-patch assertion, (3) run the loop.
        # To do this reliably, find the last `for` loop in the test code and
        # inject our wrapper just before it.
        test_body = test_code.rstrip()
        # Find last `for` at start of a line (the test-runner loop)
        for_idx = test_body.rfind("\nfor ")
        if for_idx == -1:
            # No for loop found — fall back to binary pass/fail by exec'ing
            # the full test and catching exceptions.
            script = (
                code + "\n"
                + "try:\n"
                + "    exec('''" + test_code.replace("\\", "\\\\").replace("'''", "\\'\\'\\'") + "''')\n"
                + "    print('PASS,1,1')\n"
                + "except BaseException:\n"
                + "    print('FAIL,0,1')\n"
            )
        else:
            setup = test_body[:for_idx]
            loop = test_body[for_idx:]
            script = (
                code + "\n"
                + setup + "\n"
                + counter_wrapper
                + loop + "\n"
                + "print(f'{__plus_passed[0]},{__plus_total[0]}')\n"
            )

        try:
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                timeout=self.timeout_plus,
                text=True,
                preexec_fn=_limit_memory_plus,
                env={"PATH": os.environ.get("PATH", "")},
            )
            stdout = result.stdout.strip().splitlines()
            if not stdout:
                return False, 0, 0
            last = stdout[-1]
            parts = last.split(",")
            if len(parts) != 3 and len(parts) != 2:
                return False, 0, 0
            # Handle both "PASS,1,1"/"FAIL,0,1" and "p,t" formats
            if len(parts) == 3:
                n_passed, n_total = int(parts[1]), int(parts[2])
            else:
                n_passed, n_total = int(parts[0]), int(parts[1])
            passed = (n_passed == n_total) and n_total > 0
            return passed, n_passed, n_total
        except subprocess.TimeoutExpired:
            return False, 0, 0
        except Exception:
            return False, 0, 0

    def score_plus(self, prompt, response):
        """Binary score using full augmented test suite. 1.0 if all pass, else 0.0."""
        entry = self._lookup_entry(prompt)
        if entry is None:
            return 0.0
        code = self._extract_code(response)
        if not code:
            return 0.0
        passed, _, _ = self._execute_full_test(code, entry)
        return 1.0 if passed else 0.0

    def score_plus_float(self, prompt, response):
        """Continuous score in [-1, 1] from fraction of augmented tests passed."""
        entry = self._lookup_entry(prompt)
        if entry is None:
            return -1.0
        code = self._extract_code(response)
        if not code:
            return -1.0
        _, n_passed, n_total = self._execute_full_test(code, entry)
        if n_total == 0:
            return -1.0
        return 2.0 * (n_passed / n_total) - 1.0

    def score_plus_both(self, prompt, response):
        """Return (binary, float) from a single full-test execution."""
        entry = self._lookup_entry(prompt)
        if entry is None:
            return 0.0, -1.0
        code = self._extract_code(response)
        if not code:
            return 0.0, -1.0
        passed, n_passed, n_total = self._execute_full_test(code, entry)
        binary = 1.0 if passed else 0.0
        if n_total == 0:
            return binary, -1.0
        float_score = 2.0 * (n_passed / n_total) - 1.0
        return binary, float_score

    def score_responses_plus(self, prompts, responses, batch_size=None):
        return [self.score_plus(p, r) for p, r in zip(prompts, responses)]

    def score_responses_plus_both(self, prompts, responses, batch_size=None):
        binary_scores = []
        float_scores = []
        for p, r in zip(prompts, responses):
            b, f = self.score_plus_both(p, r)
            binary_scores.append(b)
            float_scores.append(f)
        return binary_scores, float_scores
