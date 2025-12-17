"""Tests for the notebook unittest helper."""

import unittest

import pytest

from ab_tools.notebook import ab_unittest


@pytest.fixture(autouse=True)
def render_capture(monkeypatch):
    """Capture HTML output so we can make assertions."""
    rendered: list[str] = []

    def fake_html(content: str) -> str:
        rendered.append(content)
        return content

    def fake_display(content: str) -> None:
        rendered.append(content)

    monkeypatch.setattr(ab_unittest, "HTML", fake_html)
    monkeypatch.setattr(ab_unittest, "display", fake_display)
    return rendered


def test_run_unittests_html_handles_failures(render_capture):
    class MixedCase(unittest.TestCase):
        def test_pass(self):
            self.assertTrue(True)

        def test_fail(self):
            self.fail("boom")

        def test_error(self):
            raise RuntimeError("bad news")

        @unittest.skip("skip me")
        def test_skip(self):
            self.fail("should not run")

    ok = ab_unittest.run_unittests_html(
        MixedCase,
        title="Mixed results",
        show_passes=False,
    )

    assert ok is False
    html = render_capture[0]
    assert "Failures: 1" in html
    assert "Errors: 1" in html
    assert "Skipped: 1" in html
    assert 'class="row pass"' not in html


def test_run_unittests_html_filters_with_pattern(render_capture):
    class FilterCase(unittest.TestCase):
        def test_selected_case(self):
            self.assertTrue(True)

        def test_other_case(self):
            self.assertTrue(True)

    render_capture.clear()
    ok = ab_unittest.run_unittests_html(
        FilterCase,
        pattern="selected",
        title="Filtered",
    )

    assert ok is True
    html = render_capture[0]
    assert "Ran <b>1</b> tests" in html
    assert "Passed: 1" in html


def test_iter_test_cases_yields_nested_cases():
    class NestedCase(unittest.TestCase):
        def test_one(self):
            self.assertTrue(True)

        def test_two(self):
            self.assertTrue(True)

    suite = unittest.TestSuite()
    inner = unittest.TestSuite()
    inner.addTest(NestedCase("test_one"))
    suite.addTest(inner)
    suite.addTest(NestedCase("test_two"))

    cases = list(ab_unittest._iter_test_cases(suite))
    names = [c._testMethodName for c in cases]
    assert names == ["test_one", "test_two"]
