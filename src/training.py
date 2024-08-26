from dspy import Example
from csv import DictReader


def load_training_data_as_example(name: str) -> list[Example]:
    output = []
    with open(f"src/data/{name}") as f:
        file = DictReader(f)

        for row in file:
            output.append(Example(**row))

    return output
