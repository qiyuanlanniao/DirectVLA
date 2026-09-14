# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class EmbodimentTag(Enum):
    NEW_EMBODIMENT = "new_embodiment"


EMBODIMENT_TAG_MAPPING = {EmbodimentTag.NEW_EMBODIMENT.value: 31}
ROBOT_TYPE_TO_EMBODIMENT_TAG = {}
