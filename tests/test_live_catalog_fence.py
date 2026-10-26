"""The conftest fence keeps unit tests off the live model catalogs.

A slow openrouter.ai or models.dev lookup once ran tests/run_agent/test_run_agent.py
past the per-file CI limit; these tests fail if the fence stops applying.
"""
import socket

import pytest


@pytest.mark.parametrize("host", ["openrouter.ai", "models.dev", "api.openrouter.ai", b"models.dev", "OpenRouter.AI."])
def test_live_catalog_hosts_are_refused(host):
    with pytest.raises(socket.gaierror, match="test fence"):
        socket.getaddrinfo(host, 443)


@pytest.mark.parametrize("host", ["localhost", "notopenrouter.ai", "models.dev.example"])
def test_other_hosts_are_not_fenced(host):
    try:
        socket.getaddrinfo(host, 443)
    except socket.gaierror as exc:
        assert "test fence" not in str(exc)
