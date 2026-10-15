import pytest
from scripts.fix_torch_uuid_count import MARKER, PATCH, patched_source


def test_pinned_source_only_adds_uuid_fallback_and_is_idempotent():
    source = 'def _device_count_nvml():\n' + MARKER + '    return 0\n'
    result = patched_source(source)
    assert result == source.replace(MARKER,PATCH+MARKER)
    assert patched_source(result) == result


def test_unrecognized_source_is_rejected():
    with pytest.raises(ValueError,match='unrecognized'):
        patched_source('def other():\n' + MARKER)
