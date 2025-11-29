# encoding=utf-8
import logging
import os
import traceback

import cv2
import numpy as np
from ThirdPart.Segmentation.DR_anal import analyse_DR


class AnalysisWrapper:
    def __init__(self, from_path: str, static_name: str):
        self.obj_list = ["background", "hemorrhages", "hard_exudates", "microaneurysms", "disc", "soft_exudates"]
        self.lesion_mapping = {"hemorrhages": 1, "hard_exudates": 2, "microaneurysms": 3, "soft_exudates": 5}
        self.static_name = static_name
        self.from_path = from_path

    def _generate_dr_statics(self):
        img_lst = []
        for obj in self.obj_list:
            img_lst.append(cv2.imread(os.path.join(self.from_path, obj, self.static_name), 0))
        img_arr = np.array(img_lst, np.uint8)
        lesion_dct = analyse_DR(img_arr, self.lesion_mapping)

        img_name, _ = os.path.splitext(self.static_name)
        for lesion, att in lesion_dct.items():
            if not os.path.exists(os.path.join(self.from_path, "statistics", lesion)):
                os.makedirs(os.path.join(self.from_path, "statistics", lesion))
            f = open(os.path.join(self.from_path, "statistics", lesion, img_name + ".csv"), "w")
            f.write("x,y,S\n")
            for obj in lesion_dct[lesion]["objects"]:
                f.write("%d,%d,%d\n" % (obj[0], obj[1], obj[2]))  # x, y, S
            f.close()
            f = open(os.path.join(self.from_path, "statistics", lesion, "summary.csv"), "w")
            f.write("img_name,N,S\n")
            f.write("%s,%d,%f\n" % (img_name, lesion_dct[lesion]["counts"], lesion_dct[lesion]["avgS"]))
            f.close()

    def generate_dr_statics(self):
        try:
            self._generate_dr_statics()
            return True
        except Exception as e:
            logging.exception(traceback.format_exc())
            return False