import logging
import warnings

from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window
from rasterio.windows import from_bounds as window_from_bounds
import mercantile
import rasterio


__version__ = '1.4.2'

base_kwds = None
src = None

TILES_CRS = 'EPSG:3857'

log = logging.getLogger(__name__)


def init_worker(path, profile, resampling_method):
    global base_kwds, src, resampling
    resampling = Resampling[resampling_method]
    src = rasterio.open(path)
    base_kwds = profile.copy()


def process_tile(tile):
    """Process a single MBTiles tile

    Parameters
    ----------
    tile : mercantile.Tile

    Returns
    -------

    tile : mercantile.Tile
        The input tile.
    bytes : bytearray
        Image bytes corresponding to the tile.

    """
    global base_kwds, resampling, src

    # Get the bounds of the tile.
    ulx, uly = mercantile.xy(
        *mercantile.ul(tile.x, tile.y, tile.z))
    lrx, lry = mercantile.xy(
        *mercantile.ul(tile.x + 1, tile.y + 1, tile.z))

    kwds = base_kwds.copy()
    kwds['transform'] = transform_from_bounds(ulx, lry, lrx, uly,
                                              kwds['width'], kwds['height'])
    src_nodata = kwds.pop('src_nodata', None)
    dst_nodata = kwds.pop('dst_nodata', None)

    warnings.simplefilter('ignore')

    with MemoryFile() as memfile:

        with memfile.open(**kwds) as tmp:

            reproject(rasterio.band(src, tmp.indexes),
                      rasterio.band(tmp, tmp.indexes),
                      src_nodata=src_nodata,
                      dst_nodata=dst_nodata,
                      num_threads=1,
                      resampling=resampling)

        return tile, memfile.read()
