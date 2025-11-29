# encoding=utf-8
import logging
from enum import IntEnum, Enum


class USE_ENHANCE(IntEnum):
    USE = 0
    NO_USE = 1


class SEG_NAME(IntEnum):
    MB_CAMW = 0
    MB_UNET = 1


class GRADE_NAME(IntEnum):
    MB_GRADE = 0


class RUNNING_JSON_OPTIONS(Enum):
    STR_ENHANCE = "enhance"
    STR_SEG = "seg"
    STR_GRADE = "grade"


def check_running_options(json_options: dict):
    try:
        use_enhance = json_options[RUNNING_JSON_OPTIONS.STR_ENHANCE.value]
        use_seg = json_options[RUNNING_JSON_OPTIONS.STR_SEG.value]
        use_grade = json_options[RUNNING_JSON_OPTIONS.STR_GRADE.value]

        if use_enhance not in [member.value for member in USE_ENHANCE]:
            return False, "enhance param error"

        if use_seg not in [member.value for member in SEG_NAME]:
            return False, "segmentation param error"

        if use_grade not in [member.value for member in GRADE_NAME]:
            return False, "grade param error"

        return True, ""
    except Exception as e:
        return False, "Not found {}".format(str(e))
