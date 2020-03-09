import sys
import pytest
import importlib

# still got love for py2
MODULE_TYPE = type(sys)
# due to @paulganssle
@pytest.fixture(scope='function')
def clean_import():

    # remove package modules from sys.modules
    pk_mods = {mod_name: mod for mod_name, mod in sys.modules.items()
               if mod_name.startswith('package')}

    # stash names of no `package` modules  as well so we can recreate after test is ran w/o adding
    # artifiacts of the test
    other_modules  =  {mod_name for mod_name in sys.modules 
                       if mod_name not in pk_mods}

    # lets remove pk_mods
    for mod_name in pk_mods:
        del sys.modules[mod_name]

    yield # run test

    # remove anything that wasnt in the original sys.modules
    for mod_name in list(sys.modules):
        if mod_name not in other_modules:
            del sys.modules[mod_name]

    # restore pk_mods
    for mod_name, mod in pk_mods.items():
        sys.modules[mod_name] = mod


@pytest.mark.parametrize('mod', ['foo', 'bar'])
def test_lazy_import(clean_import, mod):
    
    import importlib, package
    # the mod of module type
    mod_obj = getattr(package, mod, None)
    assert isinstance(mod_obj, MODULE_TYPE)

    # is the module the same as the module we import directly
    imported_mod = importlib.import_module('package.%s' % mod)
    assert imported_mod is mod_obj

