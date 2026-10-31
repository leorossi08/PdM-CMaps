"""Pytest configuration and global hooks."""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "requirement(id): Traceability marker for plan requirement IDs")
    config.addinivalue_line("markers", "docs: Documentation synchronization and freshness checks")
    config.addinivalue_line("markers", "slow: Slow end-to-end pipeline execution checks")
