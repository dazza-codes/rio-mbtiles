# Mbtiles command.

import logging
import math
import os
import sqlite3

import click
import mercantile
import rasterio
from rasterio import MemoryFile
from rasterio.enums import Resampling
from rasterio.rio.helpers import resolve_inout
from rasterio.rio.options import output_opt
from rasterio.rio.options import overwrite_opt

from mbtiles import __version__ as mbtiles_version
from mbtiles import get_src_bounds_in_wgs84
from mbtiles import get_src_tiles
from mbtiles import get_src_zooms
from mbtiles import reproject_tile
from mbtiles import src_has_data
from mbtiles import TILES_CRS  # "EPSG:3857" Web Mercator projection

RESAMPLING_METHODS = [method.name for method in Resampling]


def validate_nodata(dst_nodata, src_nodata, meta_nodata):
    """Raise BadParameter if we don't have a src nodata for a dst"""
    if dst_nodata is not None and (src_nodata is None and meta_nodata is None):
        raise click.BadParameter(
            "--src-nodata must be provided because " "dst-nodata is not None."
        )


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
    help="A min..max range of export zoom levels. "
    "The default zoom level is the one at which "
    "the dataset is contained within a single tile.",
)
@click.option(
    "--image-dump",
    metavar="PATH",
    help="A directory into which image tiles will be optionally dumped.",
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
@click.option("--rgba", default=False, is_flag=True, help="Select RGBA output. For PNG only.")
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
    log = logging.getLogger(__name__)

    output, files = resolve_inout(files=files, output=output, overwrite=overwrite)
    input_file = files[0]

    # Name and description.
    title = title or os.path.basename(input_file)
    description = description or input_file

    # Parameters for creation of tile images.
    img_ext = "jpg" if img_format.lower() == "jpeg" else "png"

    if rgba:
        if img_format == "JPEG":
            raise click.BadParameter("RGBA output is not possible with JPEG format.")
        else:
            count = 4
    else:
        count = 3

    with ctx.obj["env"]:

        # Read metadata from the source dataset.
        # with rasterio.open(input_file) as src:
        with open(input_file, 'rb') as f, MemoryFile(f) as mem_file:
            with mem_file.open() as src:

                validate_nodata(dst_nodata, src_nodata, src.profile.get("nodata"))
                base_kwds = {"dst_nodata": dst_nodata, "src_nodata": src_nodata}

                if src_nodata is not None:
                    base_kwds.update(nodata=src_nodata)

                if dst_nodata is not None:
                    base_kwds.update(nodata=dst_nodata)

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

                # Compute the WGS84 geographic bounding box of the dataset; this
                # is used to calculate zooms and to set the MBTiles metadata:bounds.
                west, south, east, north = get_src_bounds_in_wgs84(src)

                # Resolve the minimum and maximum zoom levels for export.
                if zoom_levels:
                    min_zoom, max_zoom = map(int, zoom_levels.split(".."))
                else:
                    min_zoom, max_zoom = get_src_zooms(west, south, east, north)

                log.info("Zoom range: %d..%d", min_zoom, max_zoom)

                # Initialize the sqlite db.
                if os.path.exists(output):
                    os.unlink(output)

                # workaround for bug https://bugs.python.org/issue27126; this is only a problem
                # with libsqlite3 on OSX (< Sierra ?) with multiprocessing
                sqlite3.connect(":memory:").close()

                conn = sqlite3.connect(output)
                cur = conn.cursor()
                cur.execute("CREATE TABLE metadata (name text, value text);")
                cur.execute(
                    "CREATE TABLE tiles "
                    "(zoom_level integer, tile_column integer, tile_row integer, tile_data blob);"
                )
                conn.commit()

                insert_meta = "INSERT INTO metadata (name, value) VALUES (?, ?);"
                insert_tile = (
                    "INSERT INTO tiles "
                    "(zoom_level, tile_column, tile_row, tile_data) VALUES (?, ?, ?, ?);"
                )

                # Insert mbtiles metadata, see spec details in
                # https://github.com/mapbox/mbtiles-spec/blob/master/1.1/spec.md#metadata
                # Insert required key:value pairs
                cur.execute(insert_meta, ("name", title))
                cur.execute(insert_meta, ("type", layer_type))
                cur.execute(insert_meta, ("version", "1.1"))
                cur.execute(insert_meta, ("description", description))
                cur.execute(insert_meta, ("format", img_ext))
                # One row in metadata is suggested and, if provided, may enhance performance.
                # bounds: The maximum extent of the rendered map area. Bounds must define an
                # area covered by zoom levels. The bounds are represented in WGS84 (EPSG:4326)
                # latitude and longitude values, in the OpenLayers Bounds format
                #
                # - left, bottom, right, top
                #
                # Example of the full earth: -180.0,-85,180,85.
                cur.execute(insert_meta, ("bounds", "%f,%f,%f,%f" % (west, south, east, north)))
                conn.commit()

                # Initialize iterator over output tiles.
                tiles = get_src_tiles(west, south, east, north, min_zoom, max_zoom)

                # Check whether the input dataset contains any data
                has_data = src_has_data(src)

                for tile in tiles:
                    # MBTiles have a different origin than Mercantile.
                    mbtile_y = int(math.pow(2, tile.z)) - tile.y - 1

                    if not has_data:
                        # Insert empty tile-data into db so the tileset has tile-records.
                        cur.execute(insert_tile, (tile.z, tile.x, mbtile_y, bytearray()))
                        conn.commit()
                        continue

                    # TODO: if the min_zoom requested is a larger Tile than the min-zoom for
                    #  the src, is it possible to simply zero-pad the min-Tile without doing
                    #  a full projection every time?  Probably not, because the pixel
                    #  resolution changes with every zoom level.

                    tile_data = reproject_tile(
                        src, tile, reproject_args=base_kwds, resampling_method=resampling
                    )
                    if tile_data is None:
                        # This tile lands in an area of the src that has no data.
                        # Insert empty tile into db so the tileset has useful zoom levels.
                        cur.execute(insert_tile, (tile.z, tile.x, mbtile_y, bytes()))
                        conn.commit()
                        continue

                    # Optional image dump.
                    if image_dump:
                        img_name = "%d-%d-%d.%s" % (tile.x, mbtile_y, tile.z, img_ext)
                        img_path = os.path.join(image_dump, img_name)
                        with open(img_path, "wb") as img:
                            img.write(tile_data)

                    # Insert tile into db.
                    tile_record = (tile.z, tile.x, mbtile_y, sqlite3.Binary(tile_data))
                    cur.execute(insert_tile, tile_record)
                    conn.commit()

                conn.close()
