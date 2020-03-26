"""Module tests"""
import math

import mercantile
import numpy as np
import rasterio
from mercantile import Tile
import pytest
from rasterio import MemoryFile
from rasterio.warp import transform

import mbtiles


# TODO: is the Tile(36, 73, 7) outside the bounds of the RGB input file?


@pytest.mark.parametrize("tile", [Tile(36, 73, 7), Tile(0, 0, 0), Tile(1, 1, 1)])
@pytest.mark.parametrize("filename", ["RGB.byte.tif", "RGBA.byte.tif"])
def test_reproject_tile(data, filename: str, tile: Tile):
    src_path = str(data.join(filename))

    resampling_method = "nearest"

    dst_profile = {
        "driver": "PNG",
        "dtype": "uint8",
        "nodata": 0,
        "height": 256,
        "width": 256,
        "count": 3,
        "crs": "EPSG:3857",
    }

    # src_tiles = [Tile(x=17, y=27, z=6), Tile(x=18, y=27, z=6), Tile(x=35, y=54, z=7),
    #              Tile(x=35, y=55, z=7), Tile(x=36, y=54, z=7), Tile(x=36, y=55, z=7)]

    with rasterio.open(src_path) as src:

        # TODO: determine whether src bounds fit inside tile bounds?
        # check tile is in tiles
        # tile_applies = tile in get_src_tiles(src)
        # mercantile.bounds(tile)
        # LngLatBbox(west=-78.75, south=-27.059125784374054, east=-75.9375,
        #            north=-24.527134822597805)
        # mercantile.bounds(Tile(0, 0, 0))
        # LngLatBbox(west=-180.0, south=-85.0511287798066, east=180.0, north=85.0511287798066)

        dst_profile["count"] = src.count
        contents = mbtiles.reproject_tile(src, tile, dst_profile, resampling_method)
        assert contents
        assert isinstance(contents, bytes)


@pytest.mark.parametrize("tile", [Tile(36, 73, 7), Tile(0, 0, 0), Tile(1, 1, 1)])
def test_reproject_tile_on_empty_src(empty_data: str, tile: Tile):
    src_path = empty_data

    dst_profile = {
        "driver": "PNG",
        "dtype": "uint8",
        "nodata": 0,
        "height": 256,
        "width": 256,
        "count": 3,
        "crs": "EPSG:3857",
    }

    with rasterio.open(src_path) as src:
        dst_profile["count"] = src.count
        contents = mbtiles.reproject_tile(src, tile, dst_profile)
        #
        # tests when using an empty image output
        #
        assert contents
        assert isinstance(contents, bytes)
        with MemoryFile(contents) as mem_file:
            with mem_file.open() as empty_src:
                assert empty_src.count == 3
                band_data = empty_src.read(1)
                assert not np.any(band_data)



# TODO: add tests for a src with some empty tiles
