from dataclasses import *
from typing import Any, Callable, Dict, List, Tuple, Union, Optional
from functools import wraps
@dataclass
class EnforceClassTyping:
    def __post_init__(self):
        for (name, field_type) in self.__annotations__.items():
            if not isinstance(self.__dict__[name], field_type):
                current_type = type(self.__dict__[name])
                raise TypeError(f"The field `{name}` was assigned by `{current_type}` instead of `{field_type}`")
        # print("Check is passed successfully")
def enforce_method_typing(func: Callable) -> Callable:
    'Enforces type annotation/hints for class mathods'
    arg_annotations = func.__annotations__
    if not arg_annotations:
        return func

    @wraps(func)
    def wrapper(self, *args: Tuple[Any], **kwargs: Dict[str, Any]) -> Any:
        for arg, annotation in zip(args, arg_annotations.values()):
            if not isinstance(arg, annotation):
                raise TypeError(f"Expected {annotation} for argument {arg}, got {type(arg)}.")

        for arg_name, arg_value in kwargs.items():
            if arg_name in arg_annotations:
                annotation = arg_annotations[arg_name]
                if not isinstance(arg_value, annotation):
                    raise TypeError(f"Expected {annotation} for keyword argument {arg_name}, got {type(arg_value)}.")

        return func(self, *args, **kwargs)

    return wrapper
def enforce_function_typing(func: Callable) -> Callable:
    'Enforces type annotation/hints for other functions'
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check positional arguments
        for arg, annotation in zip(args, func.__annotations__.values()):
            if not isinstance(arg, annotation):
                raise TypeError(f"Expected {annotation} for {arg}, got {type(arg)}.")

        # Check keyword arguments
        for arg_name, arg_value in kwargs.items():
            if arg_name in func.__annotations__:
                annotation = func.__annotations__[arg_name]
                if not isinstance(arg_value, annotation):
                    raise TypeError(f"Expected {annotation} for {arg_name}, got {type(arg_value)}.")

        return func(*args, **kwargs)

    return wrapper
