# %%
from packaging.specifiers import SpecifierSet
from packaging.version import Version

specifier = SpecifierSet(">=3.1.0")

version = Version(3)

print(version in specifier)

#%%
if None: 
    print("None is True")
    print("None is False")

if not None:
    print("not None is True")
