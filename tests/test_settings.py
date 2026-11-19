"""Tests for the settings seam in chat_with_pdf.py.

Run with: pip install -r requirements.txt pytest && python -m pytest tests
These tests need no Couchbase cluster or NVIDIA API key.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import chat_with_pdf

APP = str(Path(__file__).resolve().parent.parent / "chat_with_pdf.py")


def test_find_missing_settings_reports_unset_and_empty_in_order():
    environ = {"DB_BUCKET": "pdf-docs", "DB_SCOPE": "", "INDEX_NAME": "pdf_search"}
    names = ["DB_SCOPE", "DB_BUCKET", "DB_COLLECTION", "INDEX_NAME"]

    assert chat_with_pdf.find_missing_settings(names, environ) == [
        "DB_SCOPE",
        "DB_COLLECTION",
    ]


def test_load_settings_returns_values(monkeypatch):
    for name in chat_with_pdf.REQUIRED_ENV_VARS:
        monkeypatch.setenv(name, f"value-{name}")

    settings = chat_with_pdf.load_settings(chat_with_pdf.REQUIRED_ENV_VARS)

    assert settings == {
        name: f"value-{name}" for name in chat_with_pdf.REQUIRED_ENV_VARS
    }


def test_app_stops_without_login_password(monkeypatch):
    monkeypatch.delenv("LOGIN_PASSWORD", raising=False)

    at = AppTest.from_file(APP).run()

    assert [e.value for e in at.error] == [
        "Missing environment variables: LOGIN_PASSWORD. "
        "Please add them to the secrets.toml file"
    ]
    assert len(at.text_input) == 0


@pytest.mark.parametrize("missing", [["DB_SCOPE"], ["NVIDIA_API_KEY", "INDEX_NAME"]])
def test_app_lists_every_missing_setting_after_login(monkeypatch, missing):
    monkeypatch.setenv("LOGIN_PASSWORD", "secret")
    for name in chat_with_pdf.REQUIRED_ENV_VARS:
        if name in missing:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, "set")

    at = AppTest.from_file(APP).run()
    at.text_input[0].input("secret")
    at.button[0].click()
    at.run()

    assert [e.value for e in at.error] == [
        f"Missing environment variables: {', '.join(missing)}. "
        "Please add them to the secrets.toml file"
    ]
