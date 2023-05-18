import os, shutil
from typing import List


class FileHelper:

    @staticmethod
    def calc_md5(filename):
        """Calculate the MD5 hash for a file."""
        import hashlib
        md5_hash = hashlib.md5()
        with open(filename, "rb") as file:
            for chunk in iter(lambda: file.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
