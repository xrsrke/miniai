"""A list of utility functions."""

from typing import Callable, Iterable, List, Optional, Union

__all__ = ["listify", "compose"]


def listify(objs: Optional[Union[list, str, Iterable]]):
    """Turn to a list.

    Args:
        o (_type_): objects

    Returns:
        _type_: list of objects
    """
    if objs is None:
        return []
    if isinstance(objs, list):
        return objs
    if isinstance(objs, str):
        return [objs]
    if isinstance(objs, Iterable):
        return list(objs)

    return [objs]


def compose(inp, funcs: List[Callable], *args, order_key="_order", **kwargs):
    """Apply all functions.

    Args:
        inp (_type_): _description_
        funcs (_type_): _description_
        order_key (str, optional): _description_. Defaults to '_order'.

    Returns:
        _type_: _description_
    """

    def key(obj):
        return getattr(obj, order_key, 0)

    for func in sorted(listify(funcs), key=key):
        inp = func(inp, **kwargs)
    return inp
