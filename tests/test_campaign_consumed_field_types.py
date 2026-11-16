"""Neutral exact primitive validation of consumed native/protection selectors."""

from dataclasses import replace

import pytest
from test_campaign_admission import fixture_guard
from test_campaign_initialization import config, initialization_module


class OrdinaryEqualityValue:
    def __eq__(self, other):
        return True


def test_native_endpoint_selector_requires_exact_string_before_witness():
    ca, guard, reader, _, _ = fixture_guard()
    reader.observation = replace(reader.observation, endpoint_arn=OrdinaryEqualityValue())
    with pytest.raises(ca.AdmissionError):
        guard.verify()


def test_protection_table_selector_requires_exact_string_at_construction():
    ci = initialization_module()
    original = config(ci)
    with pytest.raises(ci.InitializationError):
        replace(original, table_arn=OrdinaryEqualityValue())
