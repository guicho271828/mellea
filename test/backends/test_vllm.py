import pydantic
import pytest
from typing_extensions import Annotated

from mellea import MelleaSession
from mellea.backends.vllm import LocalVLLMBackend
from mellea.backends.types import ModelOption
import mellea.backends.model_ids as model_ids
from mellea.stdlib.base import CBlock, LinearContext
from mellea.stdlib.requirement import (
    LLMaJRequirement,
    Requirement,
    ValidationResult,
    default_output_to_bool,
)


@pytest.fixture(scope="module")
def backend():
    """Shared vllm backend for all tests in this module."""
    backend = LocalVLLMBackend(
        model_id=model_ids.QWEN3_0_6B,
        # formatter=TemplateFormatter(model_id="ibm-granite/granite-4.0-tiny-preview"),
        model_options = {
            # made smaller for a testing environment with smaller gpus.
            # such an environment could possibly be running other gpu applications, including slack
            "gpu_memory_utilization":0.8,
            "max_model_len":8192,
            "max_num_seqs":8,
        },
       )
    return backend

@pytest.fixture(scope="function")
def session(backend):
    """Fresh HuggingFace session for each test."""
    session = MelleaSession(backend, ctx=LinearContext())
    yield session
    session.reset()

@pytest.mark.qualitative
def test_system_prompt(session):
    result = session.chat(
        "Where are we going?",
        model_options={ModelOption.SYSTEM_PROMPT: "Talk like a pirate."},
    )
    print(result)


@pytest.mark.qualitative
def test_instruct(session):
    result = session.instruct("Compute 1+1.")
    print(result)


@pytest.mark.qualitative
def test_multiturn(session):
    session.instruct("Compute 1+1")
    beta = session.instruct(
        "Take the result of the previous sum and find the corresponding letter in the greek alphabet."
    )
    assert "β" in str(beta).lower()
    words = session.instruct("Now list five English words that start with that letter.")
    print(words)


@pytest.mark.qualitative
def test_format(session):
    class Person(pydantic.BaseModel):
        name: str
        email_address: Annotated[
            str, pydantic.StringConstraints(pattern=r"[a-zA-Z]{5,10}@example\.com")
        ]

    class Email(pydantic.BaseModel):
        to: Person
        subject: str
        body: str

    output = session.instruct(
        "Write a short email to Olivia, thanking her for organizing a sailing activity. Her email server is example.com. No more than two sentences. ",
        format=Email,
        model_options={ModelOption.MAX_NEW_TOKENS: 2**8},
    )
    print("Formatted output:")
    email = Email.model_validate_json(
        output.value
    )  # this should succeed because the output should be JSON because we passed in a format= argument...
    print(email)

    print("address:", email.to.email_address)
    assert "@" in email.to.email_address, "The @ sign should be in the meail address."
    assert email.to.email_address.endswith("example.com"), (
        "The email address should be at example.com"
    )

@pytest.mark.qualitative
def test_generate_from_raw(session):
    prompts = ["what is 1+1?", "what is 2+2?", "what is 3+3?", "what is 4+4?"]

    results = session.backend._generate_from_raw(
        actions=[CBlock(value=prompt) for prompt in prompts], generate_logs=None
    )

    assert len(results) == len(prompts)


@pytest.mark.qualitative
def test_generate_from_raw_with_format(session):
    prompts = ["what is 1+1?", "what is 2+2?", "what is 3+3?", "what is 4+4?"]

    class Answer(pydantic.BaseModel):
        name: str
        value: int

    results = session.backend._generate_from_raw(
        actions=[CBlock(value=prompt) for prompt in prompts],
        format=Answer,
        generate_logs=None,
    )

    assert len(results) == len(prompts)

    random_result = results[0]
    try:
        answer = Answer.model_validate_json(random_result.value)
    except pydantic.ValidationError as e:
        assert False, (
            f"formatting directive failed for {random_result.value}: {e.json()}"
        )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
