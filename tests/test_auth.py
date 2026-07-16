"""Tests for the single-operator login credential check (core.auth)."""

from __future__ import annotations

from unittest.mock import patch

from core import config
from core.auth import check_credentials


def test_correct_credentials_succeed(monkeypatch) -> None:
    """The configured username and password should authenticate."""
    monkeypatch.setattr(config, "MASTER_USERNAME", "operator")
    monkeypatch.setattr(config, "MASTER_PASSWORD", "secret")

    assert check_credentials("operator", "secret")


def test_wrong_password_fails() -> None:
    """A wrong password should not authenticate."""
    with (
        patch.object(config, "MASTER_USERNAME", "operator"),
        patch.object(config, "MASTER_PASSWORD", "secret"),
    ):
        assert not check_credentials("operator", "wrong")


def test_wrong_username_fails() -> None:
    """A wrong username should not authenticate."""
    with (
        patch.object(config, "MASTER_USERNAME", "operator"),
        patch.object(config, "MASTER_PASSWORD", "secret"),
    ):
        assert not check_credentials("intruder", "secret")


def test_empty_credentials_do_not_lock_out_fresh_checkout(monkeypatch) -> None:
    """Missing either environment value should disable the demo gate."""
    monkeypatch.setattr(config, "MASTER_USERNAME", "")
    monkeypatch.setattr(config, "MASTER_PASSWORD", "")

    assert check_credentials("", "")
