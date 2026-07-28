import pytest

from baccurate.adapters.llm.request import CanonicalLLMRequest


def test_canonical_request_has_stable_json_and_sha256_fingerprint():
    request = CanonicalLLMRequest(
        model="model-a",
        messages=(
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ),
        parameters={"temperature": 0, "seed": 100},
        response_schema_id="schema-v1",
    )

    assert request.serialize() == (
        '{"messages":[{"content":"system","role":"system"},'
        '{"content":"user","role":"user"}],"model":"model-a",'
        '"parameters":{"seed":100,"temperature":0},'
        '"response_schema_id":"schema-v1"}'
    )
    assert request.fingerprint == "d8afe1eb07779390f2eceb801889d73f7c5980a55cd90585d7f70df639f99a8b"


def test_canonical_request_ignores_mapping_order():
    first = CanonicalLLMRequest(
        model="model-a",
        messages=(
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ),
        parameters={"temperature": 0, "seed": 100},
        response_schema_id="schema-v1",
    )
    reordered_parameters = CanonicalLLMRequest(
        model="model-a",
        messages=(
            {"content": "system", "role": "system"},
            {"content": "user", "role": "user"},
        ),
        parameters={"seed": 100, "temperature": 0},
        response_schema_id="schema-v1",
    )
    assert reordered_parameters.fingerprint == first.fingerprint


@pytest.mark.parametrize("changed_component", ["message", "model", "parameter", "schema"])
def test_response_affecting_request_changes_have_distinct_identity(changed_component):
    request_values = {
        "model": "model-a",
        "messages": (
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ),
        "parameters": {"temperature": 0, "seed": 100},
        "response_schema_id": "schema-v1",
    }
    baseline = CanonicalLLMRequest(**request_values)

    if changed_component == "message":
        request_values["messages"] = tuple(reversed(request_values["messages"]))
    elif changed_component == "model":
        request_values["model"] = "model-b"
    elif changed_component == "parameter":
        request_values["parameters"] = {"temperature": 1, "seed": 100}
    else:
        request_values["response_schema_id"] = "schema-v2"

    assert CanonicalLLMRequest(**request_values).fingerprint != baseline.fingerprint
