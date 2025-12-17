"""Run unittest TestCase classes in a Jupyter notebook and display a styled HTML report."""

import re
import time
import unittest
from collections.abc import Iterator
from dataclasses import dataclass
from html import escape
from types import TracebackType
from typing import cast
import os
import sys
from pathlib import Path

try:  # pragma: no cover - exercised via tests when IPython is installed
    from IPython.display import HTML, display
except ImportError:  # pragma: no cover - executed in minimal environments

    def HTML(content: str) -> str:
        """Return the provided HTML content unchanged when IPython is absent."""
        return content

    def display(_: str) -> None:
        """No-op replacement for IPython.display.display."""
        # Notebook rendering isn't available; swallow the output in tests/CLIs.
        pass


@dataclass
class _CaseResult:
    """Serialized information about a single test case run."""

    test_id: str
    name: str
    status: str  # "pass" | "fail" | "error" | "skip"
    details: str = ""


ExcInfo = (
    tuple[type[BaseException], BaseException, TracebackType] | tuple[None, None, None]
)


class _CollectingResult(unittest.TestResult):
    """Collects per-case results while delegating to unittest.TestResult."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize tracking containers."""
        super().__init__(*args, **kwargs)
        self.cases: list[_CaseResult] = []
        self._started_at: float | None = None
        self._stopped_at: float | None = None

    def startTestRun(self) -> None:
        """Record the start timestamp of the run."""
        self._started_at = time.time()

    def stopTestRun(self) -> None:
        """Record the stop timestamp of the run."""
        self._stopped_at = time.time()

    def addSuccess(self, test: unittest.TestCase) -> None:
        """Store metadata for passing tests."""
        super().addSuccess(test)
        self.cases.append(_CaseResult(test.id(), str(test), "pass"))

    def addFailure(self, test: unittest.TestCase, err: ExcInfo) -> None:
        """Store metadata for failing tests."""
        super().addFailure(test, err)
        self.cases.append(
            _CaseResult(
                test.id(),
                str(test),
                "fail",
                self._exc_info_to_string(err, test),  # type: ignore[attr-defined]
            )
        )

    def addError(self, test: unittest.TestCase, err: ExcInfo) -> None:
        """Store metadata for tests that raised exceptions."""
        super().addError(test, err)
        self.cases.append(
            _CaseResult(
                test.id(),
                str(test),
                "error",
                self._exc_info_to_string(err, test),  # type: ignore[attr-defined]
            )
        )

    def addSkip(self, test: unittest.TestCase, reason: str) -> None:
        """Store metadata for skipped tests."""
        super().addSkip(test, reason)
        self.cases.append(_CaseResult(test.id(), str(test), "skip", reason))


def _iter_test_cases(suite: unittest.TestSuite) -> Iterator[unittest.TestCase]:
    """Yield every TestCase contained in (possibly nested) suites."""
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from _iter_test_cases(test)
        else:
            yield cast(unittest.TestCase, test)


def run_unittests_html(
    *testcase_classes: type[unittest.TestCase],
    pattern: str | None = None,
    title: str = "Unit Test Report",
    show_passes: bool = True,
) -> bool:
    """
    Run unittest TestCase classes in a Jupyter notebook and display a styled HTML report.

    Args:
        *testcase_classes: One or more unittest.TestCase subclasses.
        pattern: Optional regex to filter tests by test.id() or str(test).
        title: Report title.
        show_passes: If False, only show failures/errors/skips in the table.

    Returns:
        True if all tests passed (ignoring skips), else False.
    """
    # Build suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in testcase_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    # Optional filtering
    if pattern:
        rx = re.compile(pattern)
        filtered = unittest.TestSuite()
        for test_case in _iter_test_cases(suite):
            if rx.search(test_case.id()) or rx.search(str(test_case)):
                filtered.addTest(test_case)
        suite = filtered

    # Run
    result = _CollectingResult()
    start = time.time()
    suite.run(result)
    elapsed = time.time() - start

    # Summaries
    total = result.testsRun
    fails = len(result.failures)
    errs = len(result.errors)
    skips = len(result.skipped)
    passed = sum(1 for c in result.cases if c.status == "pass")
    ok = fails == 0 and errs == 0

    # Build rows
    def badge(status: str) -> str:
        cls = {
            "pass": "badge pass",
            "fail": "badge fail",
            "error": "badge error",
            "skip": "badge skip",
        }[status]
        label = status.upper()
        return f'<span class="{cls}">{label}</span>'

    rows_html = []
    for c in result.cases:
        if (not show_passes) and c.status == "pass":
            continue

        detail_block = ""
        if c.details:
            detail_block = f"""
            <details class="details">
              <summary>details</summary>
              <pre>{escape(c.details)}</pre>
            </details>
            """

        rows_html.append(
            f"""
          <tr class="row {c.status}">
            <td class="status">{badge(c.status)}</td>
            <td class="name">
              <div class="testname">{escape(c.name)}</div>
              <div class="testid">{escape(c.test_id)}</div>
              {detail_block}
            </td>
          </tr>
        """
        )

    table_html = (
        "\n".join(rows_html)
        if rows_html
        else """
      <tr><td colspan="2" class="empty">No tests matched your filter.</td></tr>
    """
    )

    # Render HTML
    report = f"""
    <div class="ut-wrap">
      <div class="ut-header">
        <div>
          <div class="ut-title">{escape(title)}</div>
          <div class="ut-sub">Ran <b>{total}</b> tests in <b>{elapsed:.3f}s</b></div>
        </div>
        <div class="ut-summary">
          <div class="pill pass">Passed: {passed}</div>
          <div class="pill fail">Failures: {fails}</div>
          <div class="pill error">Errors: {errs}</div>
          <div class="pill skip">Skipped: {skips}</div>
        </div>
      </div>

      <div class="ut-overall {("ok" if ok else "bad")}">
        {"✅ All tests passed" if ok else "❌ Some tests failed"}
      </div>

      <table class="ut-table">
        <thead><tr><th>Status</th><th>Test</th></tr></thead>
        <tbody>
          {table_html}
        </tbody>
      </table>
    </div>

    <style>
      .ut-wrap {{
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
        border: 1px solid #e5e7eb; border-radius: 14px; padding: 14px; margin: 8px 0;
        background: #fff;
      }}
      .ut-header {{ display:flex; justify-content:space-between; gap: 12px; align-items:flex-start; flex-wrap:wrap; }}
      .ut-title {{ font-size: 18px; font-weight: 700; }}
      .ut-sub {{ color: #6b7280; margin-top: 2px; }}
      .ut-summary {{ display:flex; gap: 8px; flex-wrap:wrap; }}
      .pill {{ padding: 6px 10px; border-radius: 999px; font-size: 12px; border: 1px solid #e5e7eb; }}
      .pill.pass {{ background: #ecfdf5; }}
      .pill.fail {{ background: #fef2f2; }}
      .pill.error {{ background: #fff7ed; }}
      .pill.skip {{ background: #eff6ff; }}

      .ut-overall {{
        margin: 12px 0 10px; padding: 10px 12px; border-radius: 12px;
        border: 1px solid #e5e7eb; font-weight: 600;
      }}
      .ut-overall.ok {{ background: #ecfdf5; }}
      .ut-overall.bad {{ background: #fef2f2; }}

      .ut-table {{ width: 100%; border-collapse: collapse; }}
      .ut-table th, .ut-table td {{ padding: 10px; border-top: 1px solid #f1f5f9; vertical-align: top; }}
      .ut-table th {{ text-align:left; font-size: 12px; color: #6b7280; letter-spacing: .02em; text-transform: uppercase; }}
      .row.pass {{ background: #ffffff; }}
      .row.fail {{ background: #fffafa; }}
      .row.error {{ background: #fff7ed; }}
      .row.skip {{ background: #f8fafc; }}

      .badge {{
        display:inline-block; font-size: 11px; font-weight: 700;
        padding: 4px 8px; border-radius: 999px; border: 1px solid #e5e7eb;
      }}
      .badge.pass {{ background: #ecfdf5; }}
      .badge.fail {{ background: #fef2f2; }}
      .badge.error {{ background: #fff7ed; }}
      .badge.skip {{ background: #eff6ff; }}

      .testname {{ font-weight: 650; }}
      .testid {{ margin-top: 2px; font-size: 12px; color:#6b7280; }}
      details.details {{ margin-top: 8px; }}
      details.details summary {{ cursor: pointer; color:#374151; font-weight:600; }}
      details.details pre {{
        margin-top: 8px; padding: 10px; border-radius: 10px;
        background: #0b1020; color: #e5e7eb; overflow-x:auto; border: 1px solid #111827;
      }}
      td.status {{ width: 110px; }}
      .empty {{ color:#6b7280; padding: 14px; }}
    </style>
    """

    display(HTML(report))
    return ok


def _run_suite_html(
    suite: unittest.TestSuite,
    pattern: str | None = None,
    title: str = "Unit Test Report",
    show_passes: bool = True,
) -> bool:
    """Run a prepared TestSuite and render the existing HTML report."""
    # Optional filtering (reuses your iterator)
    if pattern:
        rx = re.compile(pattern)
        filtered = unittest.TestSuite()
        for test_case in _iter_test_cases(suite):
            if rx.search(test_case.id()) or rx.search(str(test_case)):
                filtered.addTest(test_case)
        suite = filtered

    result = _CollectingResult()
    start = time.time()
    suite.run(result)
    elapsed = time.time() - start

    total = result.testsRun
    fails = len(result.failures)
    errs = len(result.errors)
    skips = len(result.skipped)
    passed = sum(1 for c in result.cases if c.status == "pass")
    ok = fails == 0 and errs == 0

    def badge(status: str) -> str:
        cls = {
            "pass": "badge pass",
            "fail": "badge fail",
            "error": "badge error",
            "skip": "badge skip",
        }[status]
        return f'<span class="{cls}">{status.upper()}</span>'

    rows_html = []
    for c in result.cases:
        if (not show_passes) and c.status == "pass":
            continue

        detail_block = ""
        if c.details:
            detail_block = f"""
            <details class="details">
              <summary>details</summary>
              <pre>{escape(c.details)}</pre>
            </details>
            """

        rows_html.append(
            f"""
          <tr class="row {c.status}">
            <td class="status">{badge(c.status)}</td>
            <td class="name">
              <div class="testname">{escape(c.name)}</div>
              <div class="testid">{escape(c.test_id)}</div>
              {detail_block}
            </td>
          </tr>
        """
        )

    table_html = (
        "\n".join(rows_html)
        if rows_html
        else """
      <tr><td colspan="2" class="empty">No tests matched your filter.</td></tr>
    """
    )

    report = f"""
    <div class="ut-wrap">
      <div class="ut-header">
        <div>
          <div class="ut-title">{escape(title)}</div>
          <div class="ut-sub">Ran <b>{total}</b> tests in <b>{elapsed:.3f}s</b></div>
        </div>
        <div class="ut-summary">
          <div class="pill pass">Passed: {passed}</div>
          <div class="pill fail">Failures: {fails}</div>
          <div class="pill error">Errors: {errs}</div>
          <div class="pill skip">Skipped: {skips}</div>
        </div>
      </div>

      <div class="ut-overall {("ok" if ok else "bad")}">
        {"✅ All tests passed" if ok else "❌ Some tests failed"}
      </div>

      <table class="ut-table">
        <thead><tr><th>Status</th><th>Test</th></tr></thead>
        <tbody>
          {table_html}
        </tbody>
      </table>
    </div>

    <style>
      .ut-wrap {{
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
        border: 1px solid #e5e7eb; border-radius: 14px; padding: 14px; margin: 8px 0;
        background: #fff;
      }}
      .ut-header {{ display:flex; justify-content:space-between; gap: 12px; align-items:flex-start; flex-wrap:wrap; }}
      .ut-title {{ font-size: 18px; font-weight: 700; }}
      .ut-sub {{ color: #6b7280; margin-top: 2px; }}
      .ut-summary {{ display:flex; gap: 8px; flex-wrap:wrap; }}
      .pill {{ padding: 6px 10px; border-radius: 999px; font-size: 12px; border: 1px solid #e5e7eb; }}
      .pill.pass {{ background: #ecfdf5; }}
      .pill.fail {{ background: #fef2f2; }}
      .pill.error {{ background: #fff7ed; }}
      .pill.skip {{ background: #eff6ff; }}

      .ut-overall {{
        margin: 12px 0 10px; padding: 10px 12px; border-radius: 12px;
        border: 1px solid #e5e7eb; font-weight: 600;
      }}
      .ut-overall.ok {{ background: #ecfdf5; }}
      .ut-overall.bad {{ background: #fef2f2; }}

      .ut-table {{ width: 100%; border-collapse: collapse; }}
      .ut-table th, .ut-table td {{ padding: 10px; border-top: 1px solid #f1f5f9; vertical-align: top; }}
      .ut-table th {{ text-align:left; font-size: 12px; color: #6b7280; letter-spacing: .02em; text-transform: uppercase; }}
      .row.pass {{ background: #ffffff; }}
      .row.fail {{ background: #fffafa; }}
      .row.error {{ background: #fff7ed; }}
      .row.skip {{ background: #f8fafc; }}

      .badge {{
        display:inline-block; font-size: 11px; font-weight: 700;
        padding: 4px 8px; border-radius: 999px; border: 1px solid #e5e7eb;
      }}
      .badge.pass {{ background: #ecfdf5; }}
      .badge.fail {{ background: #fef2f2; }}
      .badge.error {{ background: #fff7ed; }}
      .badge.skip {{ background: #eff6ff; }}

      .testname {{ font-weight: 650; }}
      .testid {{ margin-top: 2px; font-size: 12px; color:#6b7280; }}
      details.details {{ margin-top: 8px; }}
      details.details summary {{ cursor: pointer; color:#374151; font-weight:600; }}
      details.details pre {{
        margin-top: 8px; padding: 10px; border-radius: 10px;
        background: #0b1020; color: #e5e7eb; overflow-x:auto; border: 1px solid #111827;
      }}
      td.status {{ width: 110px; }}
      .empty {{ color:#6b7280; padding: 14px; }}
    </style>
    """

    display(HTML(report))
    return ok


def run_all_unittests_html(
    start_dir: str = "tests",
    top_level_dir: str | None = None,
    file_pattern: str = "test*.py",
    regex_filter: str | None = None,
    title: str = "All Unit Tests",
    show_passes: bool = True,
    add_cwd_to_syspath: bool = True,
) -> bool:
    """
    Discover and run *all* unittests (great for Colab/Jupyter).

    Typical layout:
      repo/
        my_package/
        tests/
          test_something.py

    Args:
        start_dir: directory to discover from (usually "tests")
        top_level_dir: repo root; if None uses current working directory
        file_pattern: unittest discovery pattern (default: test*.py)
        regex_filter: optional regex to filter test ids/names after discovery
        title: report title
        show_passes: include passing tests in table
        add_cwd_to_syspath: helps Colab find your local package when not installed

    Returns:
        True if all tests passed (skips ignored), else False.
    """
    if top_level_dir is None:
        top_level_dir = os.getcwd()

    # Make imports from the repo work in notebooks/colab
    if add_cwd_to_syspath and top_level_dir not in sys.path:
        sys.path.insert(0, top_level_dir)

    start_path = str(Path(top_level_dir) / start_dir)

    suite = unittest.defaultTestLoader.discover(
        start_dir=start_path,
        pattern=file_pattern,
        top_level_dir=top_level_dir,
    )

    return _run_suite_html(
        suite,
        pattern=regex_filter,
        title=title,
        show_passes=show_passes,
    )
