"""
Image message definition for DDS using CycloneDDS IDL.
"""
from dataclasses import dataclass
from cyclonedds.idl import IdlStruct
import cyclonedds.idl.types as types


@dataclass
class ImageChunk_(IdlStruct, typename="ImageChunk_"):
    height: types.uint32
    width: types.uint32
    depth: types.uint32
    chunk_index: types.uint32
    num_chunks: types.uint32
    data: types.sequence[types.uint8]
