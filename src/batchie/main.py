import pkgutil
import importlib


def create_instance(package_name: str, class_name: str, base_class: type, kwargs: dict):
    # Search recursively for the class within the given package
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

            # Create an instance of the class with the provided kwargs
            instance = cls(**kwargs)
            return instance

    raise NameError(
        f"A class named '{class_name}' was not found in the package '{package_name}'."
    )
