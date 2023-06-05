# coding:utf-8

from ctypes import POINTER, c_bool, c_int, pointer, sizeof, WinDLL, byref
from ctypes.wintypes import DWORD, LONG, LPCVOID

from win32 import win32api, win32gui
from win32.lib import win32con

from .c_structures import (
    ACCENT_POLICY,
    ACCENT_STATE,
    MARGINS,
    DWMNCRENDERINGPOLICY,
    DWMWINDOWATTRIBUTE,
    WINDOWCOMPOSITIONATTRIB,
    WINDOWCOMPOSITIONATTRIBDATA,
)


class WindowEffect:
    """ 调用windows api实现窗口效果 """

    DRAW_LEFT_BORDER = 0x20
    DRAW_TOP_BOARDER = 0x40
    DRAW_RIGHT_BOARDER = 0x80
    DRAW_BOTTOM_BOARDER = 0x100
    DRAW_ALL_BOARDER = 0x20 | 0x40 | 0x80 | 0x100

    def __init__(self):
        # 调用api
        self.user32 = WinDLL("user32")
        self.dwmapi = WinDLL("dwmapi")
        self.SetWindowCompositionAttribute = self.user32.SetWindowCompositionAttribute
        self.DwmExtendFrameIntoClientArea = self.dwmapi.DwmExtendFrameIntoClientArea
        self.DwmSetWindowAttribute = self.dwmapi.DwmSetWindowAttribute
        self.SetWindowCompositionAttribute.restype = c_bool
        self.DwmExtendFrameIntoClientArea.restype = LONG
        self.DwmSetWindowAttribute.restype = LONG
        self.SetWindowCompositionAttribute.argtypes = [
            c_int,
            POINTER(WINDOWCOMPOSITIONATTRIBDATA),
        ]
        self.DwmSetWindowAttribute.argtypes = [c_int, DWORD, LPCVOID, DWORD]
        self.DwmExtendFrameIntoClientArea.argtypes = [c_int, POINTER(MARGINS)]
        # 初始化结构体
        self.accentPolicy = ACCENT_POLICY()
        self.winCompAttrData = WINDOWCOMPOSITIONATTRIBDATA()
        self.winCompAttrData.Attribute = WINDOWCOMPOSITIONATTRIB.WCA_ACCENT_POLICY.value[0]
        self.winCompAttrData.SizeOfData = sizeof(self.accentPolicy)
        self.winCompAttrData.Data = pointer(self.accentPolicy)

    def setAcrylicEffect(self, hWnd, gradientColor: str = "F2F2F230", isEnableShadow: bool = True,
                         animationId: int = 0, shadowPos=DRAW_ALL_BOARDER):
        """ 给窗口开启Win10的亚克力效果

        Parameter
        ----------
        hWnd: int or `sip.voidptr`
            窗口句柄

        gradientColor: str
            十六进制亚克力混合色，对应rgba四个分量

        isEnableShadow: bool
            控制是否启用窗口阴影

        animationId: int
            控制磨砂动画

        shadowPos: int
            窗口阴影位置，可以是以下几种的任何一种或者他们的组合，
            例如 `WindowEffect.DRAW_LEFT_BORDER | WindowEffect.DRAW_TOP_BORDER`:
            * `WindowEffect.DRAW_LEFT_BORDER`: 左侧阴影
            * `WindowEffect.DRAW_TOP_BORDER`: 上侧阴影
            * `WindowEffect.DRAW_RIGHT_BORDER`: 右侧阴影
            * `WindowEffect.DRAW_BOTTOM_BORDER`: 下侧阴影
        """
        # 亚克力混合色
        gradientColor = (
            gradientColor[6:]
            + gradientColor[4:6]
            + gradientColor[2:4]
            + gradientColor[:2]
        )
        gradientColor = DWORD(int(gradientColor, base=16))
        # 磨砂动画
        animationId = DWORD(animationId)
        # 窗口阴影
        accentFlags = DWORD(shadowPos) if isEnableShadow else DWORD(0)
        self.accentPolicy.AccentState = ACCENT_STATE.ACCENT_ENABLE_ACRYLICBLURBEHIND.value[0]
        self.accentPolicy.GradientColor = gradientColor
        self.accentPolicy.AccentFlags = accentFlags
        self.accentPolicy.AnimationId = animationId
        # 开启亚克力
        self.SetWindowCompositionAttribute(
            int(hWnd), pointer(self.winCompAttrData))

    def setAeroEffect(self, hWnd):
        """ 给窗口开启Aero效果

        Parameter
        ----------
        hWnd: int or `sip.voidptr`
            窗口句柄
        """
        self.accentPolicy.AccentState = ACCENT_STATE.ACCENT_ENABLE_BLURBEHIND.value[0]
        # 开启Aero
        self.SetWindowCompositionAttribute(
            int(hWnd), pointer(self.winCompAttrData))

    def removeBackgroundEffect(self, hWnd):
        """ 移除背景特效效果 """
        self.accentPolicy.AccentState = ACCENT_STATE.ACCENT_DISABLED.value[0]
        self.SetWindowCompositionAttribute(
            int(hWnd), pointer(self.winCompAttrData))

    @staticmethod
    def moveWindow(hWnd):
        """ 移动窗口

        Parameter
        ----------
        hWnd: int or `sip.voidptr`
            窗口句柄
        """
        win32gui.ReleaseCapture()
        win32api.SendMessage(
            int(hWnd), win32con.WM_SYSCOMMAND, win32con.SC_MOVE +
            win32con.HTCAPTION, 0
        )

    def addShadowEffect(self, hWnd):
        """ 给窗口添加阴影

        Parameter
        ----------
        hWnd: int or `sip.voidptr`
            窗口句柄
        """
        hWnd = int(hWnd)
        self.DwmSetWindowAttribute(
            hWnd,
            DWMWINDOWATTRIBUTE.DWMWA_NCRENDERING_POLICY.value,
            byref(c_int(DWMNCRENDERINGPOLICY.DWMNCRP_ENABLED.value)),
            4,
        )
        margins = MARGINS(-1, -1, -1, -1)
        self.DwmExtendFrameIntoClientArea(hWnd, byref(margins))

    @staticmethod
    def addWindowAnimation(hWnd):
        """ 打开窗口动画效果

        Parameters
        ----------
        hWnd : int or `sip.voidptr`
            窗口句柄
        """
        hWnd = int(hWnd)
        style = win32gui.GetWindowLong(hWnd, win32con.GWL_STYLE)
        win32gui.SetWindowLong(
            hWnd,
            win32con.GWL_STYLE,
            style
            | win32con.WS_MAXIMIZEBOX
            | win32con.WS_CAPTION
            | win32con.CS_DBLCLKS
            | win32con.WS_THICKFRAME,
        )
