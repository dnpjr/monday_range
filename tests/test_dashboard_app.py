import hashlib
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.portfolio_results import PROTOCOL_ROOT, RUN_ROOT

DASHBOARD_PATH = Path(__file__).resolve().parents[1] / "dashboard.py"


def _sealed_bytes() -> dict[str, str]:
    files = sorted(RUN_ROOT.rglob("*.json"))
    files += sorted(RUN_ROOT.rglob("*.sha256"))
    files += [
        PROTOCOL_ROOT / "HOLDOUT_ACCESS.json",
        PROTOCOL_ROOT / "HOLDOUT_ACCESS.json.sha256",
    ]
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


class DashboardAppTests(unittest.TestCase):
    def test_all_five_pages_render_from_sealed_results(self):
        before = _sealed_bytes()
        app = AppTest.from_file(DASHBOARD_PATH, default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.radio[0].value, "Overview")
        self.assertEqual(app.title[0].value, "Monday Range Research")

        expected_sections = {
            "Strategy & Method": "Selected configuration",
            "Research Results": "Walk-forward validation",
            "Robustness & Uncertainty": "Bootstrap uncertainty",
            "Explore": "Canonical reference settings",
        }
        for page, expected_text in expected_sections.items():
            app.radio[0].set_value(page).run()
            self.assertEqual(len(app.exception), 0, page)
            rendered = " ".join(item.value for item in app.markdown)
            rendered += " " + " ".join(item.value for item in app.subheader)
            rendered += " " + " ".join(item.label for item in app.expander)
            self.assertIn(expected_text, rendered, page)

        self.assertEqual(before, _sealed_bytes())

    def test_explore_requires_an_explicit_run(self):
        app = AppTest.from_file(DASHBOARD_PATH, default_timeout=30).run()
        app.radio[0].set_value("Explore").run()
        self.assertEqual([button.label for button in app.button], ["Run exploratory backtest"])
        self.assertNotIn("explore_result", app.session_state)


if __name__ == "__main__":
    unittest.main()
