#### original stokes camera library ####
from pypylon import pylon
import numpy as np
import cv2
import os
from datetime import datetime


# 단순화된 노드 값 설정 함수
def set_node_value(nodemap, node_name, value):
    try:
        node = nodemap.GetNode(node_name)
        old_val = node.GetValue()
        if isinstance(value, int):
            node.SetValue(int(value))
        elif isinstance(value, float):
            node.SetValue(float(value))
        else:
            node.SetValue(str(value))
        # print(f"{node_name} set {old_val} to {value}.")
    except Exception as e:
        print(f"Failed to set {node_name}: {str(e)}")

# 단순화된 노드 값 가져오기 함수
def get_node_value(nodemap, node_name):
    try:
        node = nodemap.GetNode(node_name)
        return node.GetValue()
    except Exception as e:
        print(f"Failed to get {node_name}: {str(e)}")
        return None

def config_nodemap(nodemap, initial_exposure, initial_gain, pixel_format="BayerRG12"):
    # PixelFormat 설정
    set_node_value(nodemap, "PixelFormat", pixel_format)

    # LightSourcePreset 설정
    set_node_value(nodemap, "LightSourcePreset", "Off")

    # Balance White 설정
    set_node_value(nodemap, "BalanceWhiteAuto", "Off")
    """set_node_value(nodemap, "BalanceRatioSelector", "Red")
    set_node_value(nodemap, "BalanceRatio", 1.0)
    set_node_value(nodemap, "BalanceRatioSelector", "Green")
    set_node_value(nodemap, "BalanceRatio", 1.0)
    set_node_value(nodemap, "BalanceRatioSelector", "Blue")
    set_node_value(nodemap, "BalanceRatio", 1.0)
    """
    # Color Adjustment 설정
    """for color in ["Red", "Green", "Blue", "Yellow", "Cyan", "Magenta"]:
        set_node_value(nodemap, "ColorAdjustmentSelector", color)
        set_node_value(nodemap, "ColorAdjustmentHue", 0)
        set_node_value(nodemap, "ColorAdjustmentSaturation", 1.0)

    # Color Transformation 설정
    color_transforms = ["Gain00", "Gain01", "Gain02",
                        "Gain10", "Gain11", "Gain12",
                        "Gain20", "Gain21", "Gain22"]

    for transform in color_transforms:
        set_node_value(nodemap, "ColorTransformationValueSelector", transform)
        if transform in ["Gain00", "Gain11", "Gain22"]:
            set_node_value(nodemap, "ColorTransformationValue", 1.0)
        else:
            set_node_value(nodemap, "ColorTransformationValue", 0.0)"""

    set_node_value(nodemap, "ExposureAuto", "Off")
    #set_node_value(nodemap, "ExposureTime", "Off")
    if initial_exposure is not None:
        set_node_value(nodemap, "ExposureTime", initial_exposure)
    

    # Gain 초기 설정
    set_node_value(nodemap, "GainAuto", "Off")
    set_node_value(nodemap, "Gain", initial_gain)
    
    
    # Gamma
    # https://docs.baslerweb.com/gamma
    set_node_value(nodemap, "GammaEnable", False)
    set_node_value(nodemap, "BlackLevel", 0)
    # set_node_value(nodemap, "Gamma", 1.0)
