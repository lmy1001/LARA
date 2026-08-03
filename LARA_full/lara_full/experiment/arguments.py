from dataclasses import dataclass

@dataclass
class BaseArguments:
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            if k in self.__annotations__:
                setattr(self, k, v)

    def __repr__(self):
        items = [f"{key}={value!r}" for key, value in self.__annotations__.items()]
        return f"{self.__class__.__name__}(\n    " + ",\n    ".join(items) + "\n)"
