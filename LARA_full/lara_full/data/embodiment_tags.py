# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum


class EmbodimentTag(Enum):
    GR1 = "gr1"
    # """
    # The GR1 dataset.
    # """

    # OXE_DROID = "oxe_droid"
    # """
    # The OxE Droid dataset.
    # """

    # AGIBOT_GENIE1 = "agibot_genie1"
    # """
    # The AgiBot Genie-1 with gripper dataset.
    # """

    # G1 = "g1"
    # """
    # The G1 dataset.
    # """

    # Human = "human"
    # """
    # The human dataset.
    # """

    NEW_EMBODIMENT = "new_embodiment"
    """
    Any new embodiment for finetuning.
    """

    EGOEXO4D = "ego_exo_4d"
    TRUMANS = "trumans"
    LEVERB = "leverb"
    G1Upper = "g1_upper"
    G1Full = "g1_full"
    Agibot = "agibot"
    Sthv2 = "sthv2"
    OXEAustinBuds = "oxe_austin_buds"
    OXEAustinSailor = "oxe_austin_sailor"
    OXEAustinSirius = "oxe_austin_sirius"
    OXEBcZ = "oxe_bc_z"
    OXEBerkeleyAutolab = "oxe_berkeley_autolab"
    OXEBerkeleyCable = "oxe_berkeley_cable"
    OXEBerkeleyFanuc = "oxe_berkeley_fanuc"
    OXEBerkeleyMVP = "oxe_berkeley_mvp"
    OXEBerkeleyRPT = "oxe_berkeley_rpt"
    OXEBridgeOrig = "oxe_bridge_orig"
    OXECMUPlay = "oxe_cmu_play"
    OXECMUStretch = "oxe_cmu_stretch"
    OXEDlrEdan = "oxe_dlr_edan"
    OXEDobbe = "oxe_dobbe"
    OXEDroid = "oxe_droid"
    OXEFmb = "oxe_fmb"
    OXEFractal = "oxe_fractal"
    OXEFurniture = "oxe_furniture"
    OXEIamlab = "oxe_iamlab"
    OXEJacoPlay = "oxe_jaco_play"
    OXEKuka = "oxe_kuka"
    OXELanguageTable = "oxe_language_table"
    OXEMakeTea = "oxe_make_tea"
    OXENyuDoor = "oxe_nyu_door"
    OXENyuFranka = "oxe_nyu_franka"
    OXERoboturk = "oxe_roboturk"
    OXEStandfordHydra = "oxe_standford_hydra"
    OXETacoPLay = "oxe_taco_play"
    OXEToto = "oxe_toto"
    OXEUcsdKitchen = "oxe_ucsd_kitchen"
    OXEUtaustinMutex = "oxe_utaustin_mutex"
    OXEViola = "oxe_viola"
    Libero = "libero"

# Embodiment tag string: to projector index in the Action Expert Module
EMBODIMENT_TAG_MAPPING = {
    # EmbodimentTag.NEW_EMBODIMENT.value: 31,
    # EmbodimentTag.OXE_DROID.value: 17,
    # EmbodimentTag.AGIBOT_GENIE1.value: 26,
    # EmbodimentTag.GR1.value: 24,
    EmbodimentTag.EGOEXO4D.value: 1,
    EmbodimentTag.TRUMANS.value: 2,
    EmbodimentTag.LEVERB.value: 3,
    EmbodimentTag.G1Upper.value: 4,
    EmbodimentTag.G1Full.value: 5,
    EmbodimentTag.Agibot.value: 6,
    EmbodimentTag.Sthv2.value: 7,
    EmbodimentTag.OXEAustinBuds.value: 8,
    EmbodimentTag.OXEAustinSailor.value: 9,
    EmbodimentTag.OXEAustinSirius.value: 10,
    EmbodimentTag.OXEBcZ.value: 11,
    EmbodimentTag.OXEBerkeleyAutolab.value: 12,
    EmbodimentTag.OXEBerkeleyCable.value: 13,
    EmbodimentTag.OXEBerkeleyFanuc.value: 14,
    EmbodimentTag.OXEBerkeleyMVP.value: 15,
    EmbodimentTag.OXEBerkeleyRPT.value: 16,
    EmbodimentTag.OXEBridgeOrig.value: 17,
    EmbodimentTag.OXECMUPlay.value: 18,
    EmbodimentTag.OXECMUStretch.value: 19,
    EmbodimentTag.OXEDlrEdan.value: 20,
    EmbodimentTag.OXEDobbe.value: 21,
    EmbodimentTag.OXEDroid.value: 22,
    EmbodimentTag.OXEFmb.value: 23,
    EmbodimentTag.OXEFractal.value: 24,
    EmbodimentTag.OXEFurniture.value: 25,
    EmbodimentTag.OXEIamlab.value: 26,
    EmbodimentTag.OXEJacoPlay.value: 27,
    EmbodimentTag.OXEKuka.value: 28,
    EmbodimentTag.OXELanguageTable.value: 29,
    EmbodimentTag.OXEMakeTea.value: 30,
    EmbodimentTag.OXENyuDoor.value: 31,
    EmbodimentTag.OXENyuFranka.value: 32,
    EmbodimentTag.OXERoboturk.value: 33,
    EmbodimentTag.OXEStandfordHydra.value: 34,
    EmbodimentTag.OXETacoPLay.value: 35,
    EmbodimentTag.OXEToto.value: 36,
    EmbodimentTag.OXEUcsdKitchen.value: 37,
    EmbodimentTag.OXEUtaustinMutex.value: 38,
    EmbodimentTag.OXEViola.value: 39,
    EmbodimentTag.NEW_EMBODIMENT.value: 40,
    EmbodimentTag.GR1.value: 41,
    EmbodimentTag.Libero.value: 43,
}
