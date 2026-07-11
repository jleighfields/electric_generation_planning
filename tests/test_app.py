"""End-to-end tests driving the Shiny app in a real browser.

Follows https://shiny.posit.co/py/docs/end-to-end-testing.html

Prereq (once):   uv run playwright install chromium
Run:             uv run pytest -m e2e
Skip:            uv run pytest -m "not e2e"
"""

import pytest
from playwright.sync_api import Page, expect
from shiny.playwright import controller
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

# app.py lives one level up from this tests/ directory
app = create_app_fixture("../app.py")


@pytest.mark.e2e
def test_app_loads_with_default_inputs(page: Page, app: ShinyAppProc):
    page.goto(app.url)

    # intro renders and default control values are present
    expect(page.get_by_text("Electricity generation planning model")).to_be_visible()
    controller.InputText(page, "run_name").expect_value("rename me")
    controller.InputActionButton(page, "run_button").expect_label("create run")
    controller.InputRadioButtons(page, "min_obj").expect_selected("minimize cost")

    # results sections are hidden until a run is created
    expect(page.get_by_text("Resource capacities")).to_have_count(0)


@pytest.mark.e2e
@pytest.mark.slow
def test_create_run_shows_results(page: Page, app: ShinyAppProc):
    page.goto(app.url)

    controller.InputActionButton(page, "run_button").click()

    # a progress notification appears while the background solve runs
    expect(page.get_by_text("Optimization running")).to_be_visible(timeout=10_000)

    # solve runs in a background task (~15s); results appear when it completes
    expect(page.get_by_text("Resource capacities")).to_be_visible(timeout=120_000)

    # ...and the progress notification is cleared once results are shown
    expect(page.get_by_text("Optimization running")).to_have_count(0)
    expect(page.get_by_text("Cost metrics")).to_be_visible()
    expect(page.get_by_text("Generation metrics")).to_be_visible()
    expect(page.get_by_text("Hourly load and generation plot")).to_be_visible()

    # the Save run button becomes available after a run exists
    controller.InputActionButton(page, "save_button").expect_label("Save run")
