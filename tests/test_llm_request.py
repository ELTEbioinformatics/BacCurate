from baccurate.llm.request import CanonicalLLMRequest


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


def test_canonical_request_preserves_message_order_but_not_mapping_order():
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
    reordered_messages = CanonicalLLMRequest(
        model="model-a",
        messages=tuple(reversed(first.messages)),
        parameters=first.parameters,
        response_schema_id="schema-v1",
    )

    assert reordered_parameters.fingerprint == first.fingerprint
    assert reordered_messages.fingerprint != first.fingerprint
