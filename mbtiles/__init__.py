import logging
from pathlib import Path
from typing import Dict
from typing import NamedTuple

import mercantile
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import reproject

__version__ = "1.4.2"

TILES_CRS = "EPSG:3857"

log = logging.getLogger(__name__)


class TileData(NamedTuple):
    tile: mercantile.Tile
    img_path: Path
    mbtile_y: int


def process_tile(
    src: rasterio.DatasetReader,
    td: TileData,
    base_kwds: Dict,
    resampling: str = "nearest",
):
    """Process a single MBTiles tile

    The tile data is saved directly to the td.img_path

    Parameters
    ----------
    src : rasterio.DatasetReader
        The input source raster file
    td : TileData
        A named tuple to wrap the mercantile.Tile with it's output image path and mbtile-z
    base_kwds: Dict
        The output dataset profile, plus 'src_nodata' and 'dst_nodata' overrides;
        this dict is copied before it is mutated in the function
    resampling: str
        A resampling method
    """
    tile = td.tile
    resampling = Resampling[resampling]
    kwds = base_kwds.copy()
    src_nodata = kwds.pop("src_nodata", None)
    dst_nodata = kwds.pop("dst_nodata", None)

    # Get the bounds of the tile.
    ulx, uly = mercantile.xy(*mercantile.ul(tile.x, tile.y, tile.z))
    lrx, lry = mercantile.xy(*mercantile.ul(tile.x + 1, tile.y + 1, tile.z))

    kwds["transform"] = transform_from_bounds(
        ulx, lry, lrx, uly, kwds["width"], kwds["height"]
    )

    # TODO: use an async file open option?
    with rasterio.open(td.img_path, "w", **kwds) as dst:
        reproject(
            rasterio.band(src, dst.indexes),
            rasterio.band(dst, dst.indexes),
            src_nodata=src_nodata,
            dst_nodata=dst_nodata,
            num_threads=1,
            resampling=resampling,
        )
