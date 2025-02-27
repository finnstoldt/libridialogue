import os
from libridialogue import settings


def download(packages=None, directory=settings.LIBRISPEECH_PATH):
    # Set default packages if none are provided
    if packages is None:
        packages = ["dev-clean"]

    # Ensure directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Base URL for LibriSpeech data
    base_url = "http://www.openslr.org/resources/12/"

    # Download and extract the selected packages
    for package in packages:
        tar_filename = f"{package}.tar.gz"
        download_url = os.path.join(base_url, tar_filename)
        package_path = os.path.join(directory, package)

        if os.path.exists(package_path):
            print(f"Package {package} already exists, skipping...")
        else:
            os.system(f"wget {download_url}")
            os.system(f"tar -xzf {tar_filename}")
            os.system(f"mv LibriSpeech/{package} {directory}")
            os.system(f"rm {tar_filename}")
            os.system("rm -rf LibriSpeech")

    return [os.path.join(directory, package) for package in packages]


# Example usage:
# download_librispeech(packages=["dev-clean", "test-clean"], directory="MyLibriSpeechData")

if __name__ == "__main__":
    download()
