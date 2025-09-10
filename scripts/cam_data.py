"""
Image message definition for DDS using CycloneDDS IDL.
"""

from enum import auto
from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types

# root module import for resolving types
# import sensor_msgs

# PointCloud2_(std_msgs_msg_dds__Header_(), 0, 0, [], False, 0, 0, [], False)

@dataclass
@annotate.final
@annotate.autoid("sequential")
class Image_(idl.IdlStruct, ):
    header: 'unitree_sdk2py.idl.std_msgs.msg.dds_.Header_'
    height: types.uint32
    width: types.uint32
    encoding: types.uint8
    is_bigendian: bool
    step: types.uint32
    data: types.sequence[types.uint8]

class Time_(idl.IdlStruct, typename="builtin_interfaces.msg.dds_.Time_"):
    sec: types.int32
    nanosec: types.uint32




