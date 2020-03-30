# Mbtiles command.
import asyncio
import logging
import math
import os
import sqlite3
import tempfile
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict
from typing import List
from typing import NamedTuple

import aiosqlite
import click
import dask.delayed
import mercantile
import rasterio
from aiofile import AIOFile
from aiosqlite import Connection
from rasterio.enums import Resampling
from rasterio.rio.helpers import resolve_inout
from rasterio.rio.options import output_opt
from rasterio.rio.options import overwrite_opt
from rasterio.warp import transform

from mbtiles import __version__ as mbtiles_version
from mbtiles import process_tile
from mbtiles import TileData

DEFAULT_NUM_WORKERS = cpu_count() - 1
RESAMPLING_METHODS = [method.name for method in Resampling]

TILES_CRS = "EPSG:3857"


def validate_nodata(dst_nodata, src_nodata, meta_nodata):
    """Raise BadParameter if we don't have a src nodata for a dst"""
    if dst_nodata is not None and (src_nodata is None and meta_nodata is None):
        raise click.BadParameter(
            "--src-nodata must be provided because " "dst-nodata is not None."
        )


async def sqlite_create_mbtiles(conn: Connection):
    """
    Setup mbtiles database
    """

    query = "DROP TABLE IF EXISTS metadata;"
    await conn.execute(query)
    query = "CREATE TABLE metadata (name text, value text);"
    await conn.execute(query)

    query = "DROP TABLE IF EXISTS tiles;"
    await conn.execute(query)
    query = (
        "CREATE TABLE tiles (zoom_level integer, tile_column integer, tile_row integer, "
        "tile_data blob);"
    )
    await conn.execute(query)
    await conn.commit()


async def sqlite_insert_metadata(conn: Connection, values: List[Dict]):
    insert_metadata = "INSERT INTO metadata (name, value) VALUES (?, ?);"
    for value in values:
        insert_values = (value["name"], value["value"])
        await conn.execute(insert_metadata, insert_values)
    await conn.commit()


class TileValues(NamedTuple):
    zoom_level: int
    tile_column: int
    tile_row: int
    tile_data: memoryview


async def sqlite_insert_tile(conn: Connection, values: TileValues):
    insert_tiles = (
        "INSERT INTO tiles (zoom_level, tile_column, tile_row, tile_data)"
        "VALUES (?, ?, ?, ?);"
    )
    await conn.execute(insert_tiles, values)
    await conn.commit()


async def insert_tile(conn: Connection, td: TileData):

    async with AIOFile(str(td.img_path), "rb") as aio_img:
        contents = await aio_img.read()

    # Insert tile into db.
    tile_values = TileValues(
        zoom_level=td.tile.z,
        tile_column=td.tile.x,
        tile_row=td.mbtile_y,
        tile_data=sqlite3.Binary(contents)
    )
    await sqlite_insert_tile(conn, tile_values)


async def save_mbtiles(output: str, tile_data: List[TileData], metadata: List[Dict]):
    async with aiosqlite.connect(output) as db:
        await sqlite_create_mbtiles(db)
        await sqlite_insert_metadata(db, metadata)
        for td in tile_data:
            await insert_tile(db, td)
        await db.commit()


@click.command(short_help="Export a dataset to MBTiles.")
@click.argument(
    "files",
    nargs=-1,
    type=click.Path(resolve_path=True),
    required=True,
    metavar="INPUT [OUTPUT]",
)
@output_opt
@overwrite_opt
@click.option("--title", help="MBTiles dataset title.")
@click.option("--description", help="MBTiles dataset description.")
@click.option(
    "--overlay",
    "layer_type",
    flag_value="overlay",
    default=True,
    help="Export as an overlay (the default).",
)
@click.option(
    "--baselayer", "layer_type", flag_value="baselayer", help="Export as a base layer."
)
@click.option(
    "-f",
    "--format",
    "img_format",
    type=click.Choice(["JPEG", "PNG"]),
    default="JPEG",
    help="Tile image format.",
)
@click.option(
    "--tile-size",
    default=256,
    show_default=True,
    type=int,
    help="Width and height of individual square tiles to create.",
)
@click.option(
    "--zoom-levels",
    default=None,
    metavar="MIN..MAX",
    help="A min...max range of export zoom levels. "
    "The default zoom level "
    "is the one at which the dataset is contained within "
    "a single tile.",
)
@click.option(
    "--image-dump",
    metavar="PATH",
    help="A directory into which image tiles will be optionally " "dumped.",
)
@click.option(
    "-j",
    "num_workers",
    type=int,
    default=DEFAULT_NUM_WORKERS,
    help="Number of worker processes (default: %d)." % (DEFAULT_NUM_WORKERS),
)
@click.option(
    "--src-nodata",
    default=None,
    show_default=True,
    type=float,
    help="Manually override source nodata",
)
@click.option(
    "--dst-nodata",
    default=None,
    show_default=True,
    type=float,
    help="Manually override destination nodata",
)
@click.option(
    "--resampling",
    type=click.Choice(RESAMPLING_METHODS),
    default="nearest",
    show_default=True,
    help="Resampling method to use.",
)
@click.version_option(version=mbtiles_version, message="%(version)s")
@click.option(
    "--rgba", default=False, is_flag=True, help="Select RGBA output. For PNG only."
)
@click.pass_context
def mbtiles(
    ctx,
    files,
    output,
    overwrite,
    title,
    description,
    layer_type,
    img_format,
    tile_size,
    zoom_levels,
    image_dump,
    num_workers,
    src_nodata,
    dst_nodata,
    resampling,
    rgba,
):
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
    output, files = resolve_inout(files=files, output=output, overwrite=overwrite)
    inputfile = files[0]

    log = logging.getLogger(__name__)

    if image_dump:
        tile_path = Path(image_dump)
        tile_path.mkdir(parents=True, exist_ok=True)
    else:
        tile_directory = tempfile.TemporaryDirectory(prefix="rio_mbtiles_")
        tile_path = Path(tile_directory.name)

    log.info("Image outputs path: %s", tile_path)
    assert tile_path.exists()

    with ctx.obj["env"]:

        src = rasterio.open(inputfile)

        validate_nodata(dst_nodata, src_nodata, src.profile.get("nodata"))
        base_kwds = {"dst_nodata": dst_nodata, "src_nodata": src_nodata}

        if src_nodata is not None:
            base_kwds.update(nodata=src_nodata)

        if dst_nodata is not None:
            base_kwds.update(nodata=dst_nodata)

        # Name and description.
        title = title or os.path.basename(src.name)
        description = description or title

        # Compute the geographic bounding box of the dataset.
        (west, east), (south, north) = transform(
            src.crs, "EPSG:4326", src.bounds[::2], src.bounds[1::2]
        )

        # Resolve the minimum and maximum zoom levels for export.
        if zoom_levels:
            min_zoom, max_zoom = map(int, zoom_levels.split(".."))
        else:
            zw = int(round(math.log(360.0 / (east - west), 2.0)))
            zh = int(round(math.log(170.1022 / (north - south), 2.0)))
            min_zoom = min(zw, zh)
            max_zoom = max(zw, zh)

        log.debug("Zoom range: %d..%d", min_zoom, max_zoom)

        if rgba:
            if img_format == "JPEG":
                raise click.BadParameter(
                    "RGBA output is not possible with JPEG format."
                )
            else:
                count = 4
        else:
            count = 3

        # Parameters for creation of tile images.
        base_kwds.update(
            {
                "driver": img_format.upper(),
                "dtype": "uint8",
                "nodata": 0,
                "height": tile_size,
                "width": tile_size,
                "count": count,
                "crs": TILES_CRS,
            }
        )

        img_ext = "jpg" if img_format.lower() == "jpeg" else "png"

        # Initialize the sqlite db.
        if os.path.exists(output):
            os.unlink(output)

        # Constrain bounds.
        EPS = 1.0e-10
        west = max(-180 + EPS, west)
        south = max(-85.051129, south)
        east = min(180 - EPS, east)
        north = min(85.051129, north)

        # Initialize iterator over output tiles.
        tiles = mercantile.tiles(
            west, south, east, north, range(min_zoom, max_zoom + 1)
        )

        tile_data = []
        for tile in tiles:
            # MBTiles have a different origin than Mercantile.
            mbtile_y = int(math.pow(2, tile.z)) - tile.y - 1

            img_file_name = "%06d_%06d_%06d.%s" % (tile.z, tile.x, mbtile_y, img_ext)
            img_file_path = tile_path / img_file_name

            td = TileData(tile=tile, img_path=img_file_path, mbtile_y=mbtile_y)
            tile_data.append(td)

        tile_processes = []
        for td in tile_data:
            tp = dask.delayed(process_tile)(src, td, base_kwds, resampling)
            tile_processes.append(tp)
        dask.delayed(tile_processes).compute()

        src.close()

        log.info("Saving mbtiles to: %s", output)
        metadata_values = [
            {"name": "name", "value": title},
            {"name": "type", "value": layer_type},
            {"name": "version", "value": "1.1"},
            {"name": "description", "value": description},
            {"name": "format", "value": img_ext},
            {"name": "bounds", "value": "%f,%f,%f,%f" % (west, south, east, north)},
        ]

        main_loop = asyncio.get_event_loop()
        try:
            main_loop.run_until_complete(save_mbtiles(output, tile_data, metadata_values))
        finally:
            main_loop.stop()
            main_loop.close()
