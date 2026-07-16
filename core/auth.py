"""core/auth.py — Credential check for the single HealthPipe AI demo operator."""

from __future__ import annotations

import hmac

from core import config


def check_credentials(username: str, password: str) -> bool:
    """Return whether the supplied credentials match the configured operator.

    Authentication is effectively disabled when either credential is empty.
    This keeps a fresh development checkout usable without a local ``.env``;
    production-like demo deployments should configure both values.

    Args:
        username: Username entered in the login form.
        password: Password entered in the login form.

    Returns:
        True for a valid pair or when credentials are not configured.
    """
    if not config.MASTER_USERNAME or not config.MASTER_PASSWORD:
        return True
    return hmac.compare_digest(username, config.MASTER_USERNAME) and hmac.compare_digest(
        password, config.MASTER_PASSWORD
    )
