from datetime import timedelta

import pytest

from business_match.config import Settings
from business_match.schemas import BusinessContext, BusinessRecord, ClientInput, utcnow


async def test_business_context_http_journey(tmp_path):
    # Runs actual routes, disk persistence and structured HTTP model adapter.
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "business_context_check.py"
    spec = importlib.util.spec_from_file_location("business_context_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = await module.run_scenario(tmp_path)
    assert all(report["checks"].values()), report["checks"]


def test_context_validation_and_write_only_serialization():
    record = dict(
        id="r",
        kind="contract",
        subject="Мука",
        status="active",
        occurred_at=utcnow() - timedelta(days=1),
    )
    for patch in (
        dict(status="refunded"),
        dict(occurred_at=utcnow() + timedelta(days=1)),
        dict(occurred_at=utcnow().replace(tzinfo=None)),
    ):
        with pytest.raises(ValueError):
            BusinessRecord.model_validate({**record, **patch})
    context = BusinessContext(source_id="erp", source_sequence=1, records=[record])
    source = ClientInput(id="c", name="Company", business_context=context)
    assert "business_context" not in source.model_dump()
    assert "business_context" not in source.model_dump_json()
    with pytest.raises(ValueError):
        ClientInput(id="c", name="Company", business_context=None)
    assert ClientInput(id="c", name="Company").business_context is None
    assert (
        Settings(_env_file=None, business_signals_enabled=False).business_signals_enabled is False
    )
