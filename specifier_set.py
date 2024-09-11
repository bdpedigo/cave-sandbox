# %%
from packaging.specifiers import SpecifierSet
from packaging.version import Version

test_versions = ["1.0", "1.27", "2.0", "2.5", "2.20"]
test_versions = [Version(v) for v in test_versions]
constraints = ["<2,>=1.25.0", ">=2.17.0"]

specifiers = []
for constraint in constraints:
    specifier = SpecifierSet(constraint)
    specifiers.append(specifier)

for version in test_versions:
    print(version, all(version not in s for s in specifiers))

#%%
' or '.join(constraints)