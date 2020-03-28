# Mbtiles command.

import logging
import math
import tempfile
from multiprocessing import cpu_count
import os
import sqlite3
from pathlib import Path
from threading import Semaphore

import click
import dask.delayed
import mercantile
import rasterio
from rasterio.enums import Resampling
from rasterio.rio.helpers import resolve_inout
from rasterio.rio.options import overwrite_opt, output_opt
from rasterio.warp import transform

from mbtiles import process_tile
from mbtiles import __version__ as mbtiles_version
from mbtiles import TileData

DEFAULT_NUM_WORKERS = cpu_count() - 1
RESAMPLING_METHODS = [method.name for method in Resampling]

TILES_CRS = 'EPSG:3857'


def validate_nodata(dst_nodata, src_nodata, meta_nodata):
    """Raise BadParameter if we don't have a src nodata for a dst"""
    if dst_nodata is not None and (src_nodata is None and meta_nodata is None):
        raise click.BadParameter("--src-nodata must be provided because "
                                 "dst-nodata is not None.")


@click.command(short_help="Export a dataset to MBTiles.")
@click.argument(
    'files',
    nargs=-1,
    type=click.Path(resolve_path=True),
    required=True,
    metavar="INPUT [OUTPUT]")
@output_opt
@overwrite_opt
@click.option('--title', help="MBTiles dataset title.")
@click.option('--description', help="MBTiles dataset description.")
@click.option('--overlay', 'layer_type', flag_value='overlay', default=True,
              help="Export as an overlay (the default).")
@click.option('--baselayer', 'layer_type', flag_value='baselayer',
              help="Export as a base layer.")
@click.option('-f', '--format', 'img_format', type=click.Choice(['JPEG', 'PNG']),
              default='JPEG',
              help="Tile image format.")
@click.option('--tile-size', default=256, show_default=True, type=int,
              help="Width and height of individual square tiles to create.")
@click.option('--zoom-levels',
              default=None,
              metavar="MIN..MAX",
              help="A min...max range of export zoom levels. "
                   "The default zoom level "
                   "is the one at which the dataset is contained within "
                   "a single tile.")
@click.option('--image-dump',
              metavar="PATH",
              help="A directory into which image tiles will be optionally "
                   "dumped.")
@click.option('-j', 'num_workers', type=int, default=DEFAULT_NUM_WORKERS,
              help="Number of worker processes (default: %d)." % (
                  DEFAULT_NUM_WORKERS))
@click.option('--src-nodata', default=None, show_default=True,
              type=float, help="Manually override source nodata")
@click.option('--dst-nodata', default=None, show_default=True,
              type=float, help="Manually override destination nodata")
@click.option('--resampling', type=click.Choice(RESAMPLING_METHODS),
              default='nearest', show_default=True,
              help="Resampling method to use.")
@click.version_option(version=mbtiles_version, message='%(version)s')
@click.option('--rgba', default=False, is_flag=True, help="Select RGBA output. For PNG only.")
@click.pass_context
def mbtiles(ctx, files, output, overwrite, title, description,
            layer_type, img_format, tile_size, zoom_levels, image_dump,
            num_workers, src_nodata, dst_nodata, resampling, rgba):
    """Export a dataset to MBTiles (version 1.1) in a SQLite file.

    The input dataset may have any coordinate reference system. It must
    have at least three bands, which will be become the red, blue, and
    green bands of the output image tiles.

    An optional fourth alpha band may be copied to the output tiles by
    using the --rgba option in combination with the PNG format. This
    option requires that the input dataset has at least 4 bands.

    If no zoom levels are specified, the defaults are the zoom levels
    nearest to the one at which one tile may contain the entire source
    dataset.

    If a title or description for the output file are not provided,
    they will be taken from the input dataset's filename.

    This command is suited for small to medium (~1 GB) sized sources.

    Python package: rio-mbtiles (https://github.com/mapbox/rio-mbtiles).
    """
    output, files = resolve_inout(files=files, output=output,
                                  overwrite=overwrite)
    inputfile = files[0]

    log = logging.getLogger(__name__)

    if image_dump:
        tile_path = Path(image_dump)
        tile_path.mkdir(parents=True, exist_ok=True)
    else:
        tile_directory = tempfile.TemporaryDirectory(prefix='rio_mbtiles_')
        tile_path = Path(tile_directory.name)

    log.info("Image outputs path: %s", tile_path)
    assert tile_path.exists()

    with ctx.obj['env']:

        src = rasterio.open(inputfile)

        validate_nodata(dst_nodata, src_nodata, src.profile.get('nodata'))
        base_kwds = {'dst_nodata': dst_nodata, 'src_nodata': src_nodata}

        if src_nodata is not None:
            base_kwds.update(nodata=src_nodata)

        if dst_nodata is not None:
            base_kwds.update(nodata=dst_nodata)

        # Name and description.
        title = title or os.path.basename(src.name)
        description = description or title

        # Compute the geographic bounding box of the dataset.
        (west, east), (south, north) = transform(
            src.crs, 'EPSG:4326', src.bounds[::2], src.bounds[1::2])

        # Resolve the minimum and maximum zoom levels for export.
        if zoom_levels:
            min_zoom, max_zoom = map(int, zoom_levels.split('..'))
        else:
            zw = int(round(math.log(360.0 / (east - west), 2.0)))
            zh = int(round(math.log(170.1022 / (north - south), 2.0)))
            min_zoom = min(zw, zh)
            max_zoom = max(zw, zh)

        log.debug("Zoom range: %d..%d", min_zoom, max_zoom)

        if rgba:
            if img_format == 'JPEG':
                raise click.BadParameter("RGBA output is not possible with JPEG format.")
            else:
                count = 4
        else:
            count = 3

        # Parameters for creation of tile images.
        base_kwds.update({
            'driver': img_format.upper(),
            'dtype': 'uint8',
            'nodata': 0,
            'height': tile_size,
            'width': tile_size,
            'count': count,
            'crs': TILES_CRS})

        img_ext = 'jpg' if img_format.lower() == 'jpeg' else 'png'

        # Initialize the sqlite db.
        if os.path.exists(output):
            os.unlink(output)

        # workaround for bug here: https://bugs.python.org/issue27126
        sqlite3.connect(':memory:').close()

        sqlite_semaphore = Semaphore()
        conn = sqlite3.connect(output)
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE tiles "
            "(zoom_level integer, tile_column integer, "
            "tile_row integer, tile_data blob);")
        cur.execute(
            "CREATE TABLE metadata (name text, value text);")
        conn.commit()

        # Insert mbtiles metadata into db.
        cur.execute(
            "INSERT INTO metadata (name, value) VALUES (?, ?);",
            ("name", title))
        cur.execute(
            "INSERT INTO metadata (name, value) VALUES (?, ?);",
            ("type", layer_type))
        cur.execute(
            "INSERT INTO metadata (name, value) VALUES (?, ?);",
            ("version", "1.1"))
        cur.execute(
            "INSERT INTO metadata (name, value) VALUES (?, ?);",
            ("description", description))
        cur.execute(
            "INSERT INTO metadata (name, value) VALUES (?, ?);",
            ("format", img_ext))
        cur.execute(
            "INSERT INTO metadata (name, value) VALUES (?, ?);",
            ("bounds", "%f,%f,%f,%f" % (west, south, east, north)))

        conn.commit()

        # Constrain bounds.
        EPS = 1.0e-10
        west = max(-180 + EPS, west)
        south = max(-85.051129, south)
        east = min(180 - EPS, east)
        north = min(85.051129, north)

        # Initialize iterator over output tiles.
        tiles = mercantile.tiles(
            west, south, east, north, range(min_zoom, max_zoom + 1))

        tile_data = []
        for tile in tiles:
            # MBTiles have a different origin than Mercantile.
            mbtile_y = int(math.pow(2, tile.z)) - tile.y - 1

            img_file_name = "%06d_%06d_%06d.%s" % (tile.z, tile.x, mbtile_y, img_ext)
            img_file_path = tile_path / img_file_name

            td = TileData(
                tile=tile,
                img_path=img_file_path,
                mbtile_y=mbtile_y
            )
            tile_data.append(td)

        # TODO: this uses threads but does not release the GIL, so it's probably
        #       not as fast as the default use of a multiprocessing pool with a
        #       shared src dataset.  If possible, use a shared object memory
        #       with dask processes or somehow run the projections using a
        #       shared numpy.ndarray with threads if it releases the GIL.
        tile_processes = []
        for td in tile_data:
            tp = dask.delayed(process_tile)(src, td, base_kwds, resampling)
            tile_processes.append(tp)
        dask.delayed(tile_processes).compute()
        src.close()

        for td in tile_data:

            with open(str(td.img_path), "rb") as img:
                contents = img.read()

            # Insert tile into db.
            cur.execute(
                "INSERT INTO tiles "
                "(zoom_level, tile_column, tile_row, tile_data) "
                "VALUES (?, ?, ?, ?);",
                (td.tile.z, td.tile.x, td.mbtile_y, sqlite3.Binary(contents)))
            conn.commit()

        conn.commit()
        conn.close()
