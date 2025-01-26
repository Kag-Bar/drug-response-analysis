import unittest
import pkg_resources
import re

class TestRequirements(unittest.TestCase):

    def setUp(self):
        """Read the requirements file and extract the package and version."""
        self.requirements = {}
        with open("requirements.txt", "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line:  # Ignore empty lines
                    package, version_constraint = self._parse_requirement(line)
                    self.requirements[package] = version_constraint

    def _parse_requirement(self, requirement):
        """Parse the package and version constraint."""
        # This regex captures packages with version constraints like ==, >=, <=, etc.
        match = re.match(r"([a-zA-Z0-9_-]+)([><=]+[\d\.]+)?", requirement)
        if match:
            package = match.group(1)
            version_constraint = match.group(2) if match.group(2) else ""
            return package, version_constraint
        else:
            raise ValueError(f"Invalid requirement format: {requirement}")

    def test_requirements_installed(self):
        """Test if all required packages are installed and meet the version constraints."""
        for package, version_constraint in self.requirements.items():
            try:
                installed_version = pkg_resources.get_distribution(package).version
                if version_constraint:
                    self._check_version_constraint(package, installed_version, version_constraint)
                else:
                    self.assertIsNotNone(installed_version, f"Package {package} is installed but has no version constraint.")
            except pkg_resources.DistributionNotFound:
                self.fail(f"Package {package} is not installed")

    def _check_version_constraint(self, package, installed_version, version_constraint):
        """Check if the installed version satisfies the version constraint."""
        if version_constraint.startswith("=="):
            if installed_version != version_constraint[2:]:
                self.fail(f"Version mismatch found in library '{package}': "
                          f"Installed version is '{installed_version}', but expected version is '{version_constraint[2:]}'")
        elif version_constraint.startswith(">="):
            if installed_version < version_constraint[2:]:
                self.fail(f"Version mismatch found in library '{package}': "
                          f"Installed version is '{installed_version}', but expected version to be >= '{version_constraint[2:]}'")
        elif version_constraint.startswith("<="):
            if installed_version > version_constraint[2:]:
                self.fail(f"Version mismatch found in library '{package}': "
                          f"Installed version is '{installed_version}', but expected version to be <= '{version_constraint[2:]}'")
        elif version_constraint.startswith(">"):
            if installed_version <= version_constraint[1:]:
                self.fail(f"Version mismatch found in library '{package}': "
                          f"Installed version is '{installed_version}', but expected version to be > '{version_constraint[1:]}'")
        elif version_constraint.startswith("<"):
            if installed_version >= version_constraint[1:]:
                self.fail(f"Version mismatch found in library '{package}': "
                          f"Installed version is '{installed_version}', but expected version to be < '{version_constraint[1:]}'"
                          f"\nPlease change the library version to '{version_constraint[1:]}'")

if __name__ == "__main__":
    unittest.main()
