import pkg_resources

# Dynamically fetch package versions
def get_installed_versions(package_names):
    versions = {}
    for package in package_names:
        try:
            version = pkg_resources.get_distribution(package).version
            versions[package] = version
        except pkg_resources.DistributionNotFound:
            versions[package] = "Not Installed"
    return versions

# Define the packages you're using
PACKAGE_NAMES = [
    "python",
    "streamlit",
    "langchain",
    "faiss-cpu",
    "sentence-transformers"
]

# Get versions
PACKAGE_VERSIONS = get_installed_versions(PACKAGE_NAMES)

def print_versions():
    """Print package versions for reference."""
    for package, version in PACKAGE_VERSIONS.items():
        print(f"{package}: {version}")
