#!/usr/bin/env python3
"""init file for datafiles."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path

DIR_PATH = Path(__file__).parent
EXAMPLES = DIR_PATH / ".." / ".." / "examples"

# bulk water (NpT)
WATER_TRR_NPT = EXAMPLES / "water.trr"
WATER_TPR_NPT = EXAMPLES / "water.tpr"
