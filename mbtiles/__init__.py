import logging
import math
import warnings
from typing import Iterable
from typing import NamedTuple
from typing import Optional
from typing import Tuple

import mercantile
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import reproject
from rasterio.warp import transform
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.windows import Window

__version__ = "1.4.2"


TILES_CRS = "EPSG:3857"  # Web Mercator projection
WGS84_CRS = "EPSG:4326"  # The ellipsoidal reference for web-mercator projection

log = logging.getLogger(__name__)


class MercatorXY(NamedTuple):
    x: float
    y: float


class MercatorBoundsXY(NamedTuple):
    ul: MercatorXY
    lr: MercatorXY


def get_mercator_xy_bounds(tile: mercantile.Tile) -> MercatorBoundsXY:
    """
    Get the mercantile bounds of the tile as web mercator x, y
    """
    ul_lnglat = mercantile.ul(tile)  # upper left of tile
    # find lower-right for tile, using the upper left of "next" tile
    lr_lnglat = mercantile.ul(tile.x + 1, tile.y + 1, tile.z)
    ul_x, ul_y = mercantile.xy(lng=ul_lnglat.lng, lat=ul_lnglat.lat)
    lr_x, lr_y = mercantile.xy(lng=lr_lnglat.lng, lat=lr_lnglat.lat)
    ul = MercatorXY(x=ul_x, y=ul_y)
    lr = MercatorXY(x=lr_x, y=lr_y)
    return MercatorBoundsXY(ul=ul, lr=lr)


def get_tile_window(src: rasterio.DatasetReader, tile: mercantile.Tile) -> Optional[Window]:
    tile_bounds = get_mercator_xy_bounds(tile)
    west, south, east, north = transform_bounds(
        src_crs=TILES_CRS,
        dst_crs=src.crs,
        left=tile_bounds.ul.x,
        top=tile_bounds.ul.y,
        bottom=tile_bounds.lr.y,
        right=tile_bounds.lr.x,
    )
    tile_window = window_from_bounds(west, south, east, north, transform=src.transform)
    if np.isfinite(tile_window.width) and np.isfinite(tile_window.height):

        # TODO: create a Window for the tile regardless of the src projection?

        # adjusted_tile_window = Window(
        #     tile_window.col_off - 1,
        #     tile_window.row_off - 1,
        #     tile_window.width + 2,
        #     tile_window.height + 2,
        # )
        # tile_window = adjusted_tile_window.round_offsets().round_shape()
        return tile_window


def has_tile_data(src: rasterio.DatasetReader, tile_window: Window) -> bool:
    return src.read_masks(1, window=tile_window).any()


def get_src_bounds_in_wgs84(src) -> Tuple[float, float, float, float]:
    (west, east), (south, north) = transform(
        src.crs, WGS84_CRS, src.bounds[::2], src.bounds[1::2]
    )
    # TODO: can these bounds be packaged in a class?
    return west, south, east, north


def get_src_zooms(west, south, east, north) -> Tuple[int, int]:
    zw = int(round(math.log(360.0 / (east - west), 2.0)))
    zh = int(round(math.log(170.1022 / (north - south), 2.0)))
    min_zoom = min(zw, zh)
    max_zoom = max(zw, zh)
    return min_zoom, max_zoom


def get_src_tiles(west, south, east, north, min_zoom, max_zoom) -> Iterable[mercantile.Tile]:
    # Constrain bounds within limits of spherical mercator
    eps = 1.0e-10
    west = max(-180 + eps, west)
    south = max(-85.051129, south)
    east = min(180 - eps, east)
    north = min(85.051129, north)

    return mercantile.tiles(west, south, east, north, range(min_zoom, max_zoom + 1))


def src_has_data(src) -> bool:
    # Check whether the input dataset contains any data
    src_indexes = (1, 2, 3)  # ignore any alpha band in RGBA input
    if len(src.indexes) < 3:
        src_indexes = src.indexes
    if src.nodata is None:
        has_bands_data = [np.any(src.read(i)) for i in src_indexes]
    else:
        has_bands_data = [np.any(src.read(i) != src.nodata) for i in src_indexes]
    return any(has_bands_data)


EMPTY_TILE = None


def empty_tile(profile) -> bytes:
    width = profile.get("width", 256)
    height = profile.get("height", 256)
    tile_data = np.zeros((width, height), dtype=np.uint8)
    profile["nodata"] = 0
    profile["width"] = width
    profile["height"] = height
    with MemoryFile() as mem_file:
        with mem_file.open(**profile) as dst:
            for idx in dst.indexes:
                dst.write(tile_data, idx)
        return mem_file.read()


def reproject_tile(
    src: rasterio.DatasetReader,
    tile: mercantile.Tile,
    reproject_args: dict,
    resampling_method: str = "nearest",
) -> Optional[bytes]:
    """
    Process a single MBTiles tile

    Parameters
    ----------
    src : rasterio.DatasetReader
        The source dataset
    tile : mercantile.Tile
    reproject_args : Dict
        Common reproject arguments and target tile profile settings
    resampling_method: str

    Returns
    -------

    bytes : Optional[bytes]
        Projected image bytes for the tile
    """
    global EMPTY_TILE

    dst_profile = reproject_args.copy()
    src_nodata = dst_profile.pop("src_nodata", None)
    dst_nodata = dst_profile.pop("dst_nodata", None)

    tile_bounds = get_mercator_xy_bounds(tile)
    dst_profile["transform"] = transform_from_bounds(
        north=tile_bounds.ul.y,
        west=tile_bounds.ul.x,
        south=tile_bounds.lr.y,
        east=tile_bounds.lr.x,
        width=dst_profile["width"],
        height=dst_profile["height"],
    )

    warnings.simplefilter("ignore")

    if not src_has_data(src):
        if EMPTY_TILE is None:
            EMPTY_TILE = empty_tile(dst_profile)
        return EMPTY_TILE

    # TODO: handle the cases where the src entirely fits in the tile
    #       - run a projection

    # TODO: handle the cases where the tile is a Window inside the src
    #       - if the src-window has no data, return None
    #       - otherwise, run a projection

    # TODO: determine whether tile bounds fit inside src bounds (use s2geometry ?)
    # check tile is in tiles
    # tile_applies = tile in get_src_tiles(src)
    # mercantile.bounds(tile)
    # LngLatBbox(west=-78.75, south=-27.059125784374054, east=-75.9375,
    #            north=-24.527134822597805)
    # mercantile.bounds(Tile(0, 0, 0))
    # LngLatBbox(west=-180.0, south=-85.0511287798066, east=180.0, north=85.0511287798066)

    # tile_window = get_tile_window(src, tile)
    # if tile_window:
    #     # TODO: try the boundless option if the tile is larger than src
    #     tile_mask = src.read_masks(1, window=tile_window)
    #     if not tile_mask.any():
    #         # TODO: write out all NoData or return None?
    #         return None
    #         # with MemoryFile() as mem_file:
    #         #     with mem_file.open(**dst_profile) as dst:
    #         #         for i in dst.indexes:
    #         #             dst.write(tile_mask, i)
    #         #     return mem_file.read()

    with MemoryFile() as mem_file:
        with mem_file.open(**dst_profile) as dst:
            reproject(
                source=rasterio.band(src, dst.indexes),
                destination=rasterio.band(dst, dst.indexes),
                src_nodata=src_nodata,
                dst_nodata=dst_nodata,
                num_threads=1,
                resampling=Resampling[resampling_method],
            )
        return mem_file.read()
