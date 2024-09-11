# %%
from packaging.specifiers import SpecifierSet
from packaging.version import Version

specifier = SpecifierSet(">=3.1.0")

version = Version("3")

print(version in specifier)
