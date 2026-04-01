def get_pylogger(*args, **kwargs):
    from src.utils.pylogger import get_pylogger as _get_pylogger
    return _get_pylogger(*args, **kwargs)


def enforce_tags(*args, **kwargs):
    from src.utils.rich_utils import enforce_tags as _enforce_tags
    return _enforce_tags(*args, **kwargs)


def print_config_tree(*args, **kwargs):
    from src.utils.rich_utils import print_config_tree as _print_config_tree
    return _print_config_tree(*args, **kwargs)


from src.utils.utils import *

__all__ = ["get_pylogger", "enforce_tags", "print_config_tree"]