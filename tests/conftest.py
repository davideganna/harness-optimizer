import pytest

from harness_optimizer.core import Dataset, Example


@pytest.fixture
def tiny_dataset() -> Dataset:
    return Dataset(
        name="tiny",
        examples=[
            Example(id=str(i), inputs={"x": str(i)}, expected=str(i * 2), metadata={"parity": i % 2})
            for i in range(20)
        ],
    )
