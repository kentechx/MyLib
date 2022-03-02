import os, shutil
from typing import List

class FileHelper:

    @staticmethod
    def overwrite_files(src_fps:List[str], dst_fps:List[str]):
        src_fn_fp_dict = {os.path.split(fp)[-1]:fp for fp in src_fps}
        dst_fn_fp_dict = {os.path.split(fp)[-1]:fp for fp in dst_fps}

        for fn, dst_fp in dst_fn_fp_dict.items():
            src_fp = src_fn_fp_dict[fn]
            shutil.copy(src_fp, dst_fp)
            print('Overwrite file', dst_fp)
