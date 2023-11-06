import importlib
import inspect
import pkgutil
from typing import Dict, Any


def get_class(package_name: str, class_name: str, base_class: type) -> type:
    """
    Get a class from a package by name.

    :param package_name: The name of the package to search.
    :param class_name: The name of the class to search for.
    :param base_class: The base class that the class should inherit from.
    """
    package = importlib.import_module(package_name)
    for _, module_name, _ in pkgutil.walk_packages(
        package.__path__, package_name + "."
    ):
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name, None)
        if cls:
            # Assert that the class is a subclass of the given base class
            if not issubclass(cls, base_class):
                raise ValueError(
                    f"The class '{class_name}' is not a subclass of '{base_class.__name__}'"
                )
            return cls


def create_instance(package_name: str, class_name: str, base_class: type, kwargs: dict):
    """
    Create an instance of a class from a package by name.

    :param package_name: The name of the package to search.
    :param class_name: The name of the class to search for.
    :param base_class: The base class that the class should inherit from.
    :param kwargs: Keyword arguments to pass to the class constructor.
    """
    cls = get_class(package_name, class_name, base_class)

    if not cls:
        raise NameError(
            f"A class named '{class_name}' was not found in the package '{package_name}'."
        )

    # Create an instance of the class with the provided kwargs
    instance = cls(**kwargs)
    return instance


def get_required_init_args_with_annotations(cls) -> Dict[str, Any]:
    """
    Get a dictionary of required __init__ arguments and their type annotations for a given class.

    :param cls: The class to inspect.
    :return: A dictionary with argument names as keys and their type annotations as values.
    """
    if not inspect.isclass(cls):
        raise TypeError("The given object is not a class.")

    init_signature = inspect.signature(cls.__init__)
    parameters = init_signature.parameters
    required_args_with_annotations = {}

    for name, param in parameters.items():
        if name == "self":
            continue

        if param.default == inspect.Parameter.empty:
            annotation = (
                param.annotation
                if param.annotation != inspect.Parameter.empty
                else None
            )
            required_args_with_annotations[name] = annotation

    return required_args_with_annotations
