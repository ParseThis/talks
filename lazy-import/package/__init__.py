# from . import bar
# from . import foo

__all__ = ['foo', 'bar']

def __getattr__(module_name):
    import importlib 
    if module_name in __all__:
        return importlib.import_module('.' + module_name, __name__)
    # still got love for py2.7
    raise AttributeError('module {!r} has no attribute {!r}'.format(__name__,
                                                                    module_name))

def __dir__():
    """ recover package.__dir__ and add packages available due to lazy imports
    """
    return list(globals().keys()) + __all__ 

