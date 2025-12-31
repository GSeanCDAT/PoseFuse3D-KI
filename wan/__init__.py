from enum import Enum


class TaskType(str, Enum):
    T2V = "text2video"
    I2V = "image2video"
