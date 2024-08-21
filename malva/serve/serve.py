import io
import logging
import numpy as np
import json
import pandas as pd
import os

import threading


from netCDF4 import Dataset
import xarray as xr
import datashader as ds

from flask import (
    Flask,
    render_template,
    send_file,
    Blueprint,
    request,
    session,
    g,
    jsonify,
)
from flask_session import Session

from skimage.filters import gaussian
from PIL import Image
import uuid

from malva.index import MalvaIndex
from malva.utils import check_file_exists
from malva.dbutils import handle_sequence
from malva.serve.modeling import handle_natural_query, setup_model
from malva.serve.templates.strings import HINT_SEQUENCE_QUERY

MAX_LOAD_ALL = 100_000_000
MAX_DATA_POINTS = 100_000
SEQ_MAX_LEN = 1_000

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "BAD_SECRET_KEY")
app.config["SESSION_TYPE"] = "cachelib"
app.config["SESSION_PERMANENT"] = True
app.config["SESSION_USE_SIGNER"] = False
Session(app)

map_bp = Blueprint("map", __name__)


class GlobalState:
    def __init__(self):
        self.kmer_index = None
        self.xy = None
        self.xmax = None
        self.ymax = None
        self.xmin = None
        self.ymin = None
        self._loc_all = None
        self._abu_all = None

    def initialize(self, args):
        logging.info("Loading malva index and metadata")
        try:
            self.kmer_index = MalvaIndex(args.index_in)
            self.kmer_index.open()

            self._loc_all, self._abu_all = np.unique(
                self.kmer_index.index[f"index_0_data"][0:MAX_LOAD_ALL],
                return_counts=True,
            )

            logging.info("Loading the pointers into memory. Might take a while.")
            self.kmer_index.where(
                "A" * self.kmer_index.kmer_size + "T", max_mem=args.max_mem, use_background_model=False
            )

            self.xmax = self.kmer_index.coord_lims[1] + 1
            self.ymax = self.kmer_index.coord_lims[3] + 1
            self.xmin = self.kmer_index.coord_lims[0]
            self.ymin = self.kmer_index.coord_lims[2]
            self.xy = self.kmer_index.spatial_coord[:]
            logging.info("Loaded the pointers to memory! Will not close file...")

            logging.info("Initializing the language model")
            # setup_model()
        except Exception as e:
            logging.error(f"Error initializing global state: {str(e)}")
            raise


global_state = GlobalState()


class UserSession:
    _instances = {}
    _lock = threading.Lock()

    def __init__(self, session_id):
        self.session_id = session_id
        self.outfile_name = f"user_{self.session_id}.nc"
        self.data = None
        self.whole_max_ints = []
        self.file_lock = threading.Lock()

    @classmethod
    def get_instance(self, session_id):
        with self._lock:
            if session_id not in self._instances:
                print("create session")
                self._instances[session_id] = self(session_id)
            return self._instances[session_id]

    def generate_netcdf_index(self, render_scale=1):
        with self.file_lock:
            if self.data is not None:
                return

            if check_file_exists(self.outfile_name):
                self.data = xr.open_dataset(self.outfile_name)
                return

            # generate image size based on the scaling factor
            im_shape = np.array(
                [
                    int(
                        (
                            global_state.kmer_index.coord_lims[1]
                            - global_state.kmer_index.coord_lims[0]
                        )
                        * render_scale
                    ),
                    int(
                        (
                            global_state.kmer_index.coord_lims[3]
                            - global_state.kmer_index.coord_lims[2]
                        )
                        * render_scale
                    ),
                ]
            )

            try:
                with Dataset(self.outfile_name, "w", format="NETCDF4") as ncfile:
                    x_dim = ncfile.createDimension("x", im_shape[0])
                    y_dim = ncfile.createDimension("y", im_shape[1])

                    x = ncfile.createVariable("x", np.float32, ("x",))
                    y = ncfile.createVariable("y", np.float32, ("y",))

                    x[:] = np.linspace(
                        global_state.kmer_index.coord_lims[0],
                        global_state.kmer_index.coord_lims[1],
                        im_shape[0],
                    )
                    y[:] = np.linspace(
                        global_state.kmer_index.coord_lims[2],
                        global_state.kmer_index.coord_lims[3],
                        im_shape[1],
                    )

                    all_s = ncfile.createVariable(
                        "all",
                        "u1",
                        (
                            "x",
                            "y",
                        ),
                        fill_value=0,
                    )

                    # rasterize the data
                    xy = global_state.xy[global_state._loc_all]
                    im, _, _ = np.histogram2d(
                        xy[:, 0],
                        xy[:, 1],
                        weights=global_state._abu_all,
                        range=[
                            [
                                global_state.kmer_index.coord_lims[0],
                                global_state.kmer_index.coord_lims[1],
                            ],
                            [
                                global_state.kmer_index.coord_lims[2],
                                global_state.kmer_index.coord_lims[3],
                            ],
                        ],
                        bins=tuple(im_shape),
                    )
                    im = ((im / im.max()) * 255).astype(np.uint8)

                    all_s[:, :] = im

                self.data = xr.open_dataset(self.outfile_name)
            except Exception as e:
                logging.error(f"Error generating netCDF index: {str(e)}")
                raise

    def add_kmer_to_netcdf_index(self, locs, ints, render_scale=1):
        with self.file_lock:
            if self.data is not None:
                self.data.close()
                del self.data
                self.data = None

            _exists = check_file_exists(self.outfile_name)
            _mode = "r+" if _exists else "w"

            with Dataset(self.outfile_name, _mode, format="NETCDF4") as ncfile:
                if _exists and "ints" in ncfile.variables:
                    ints_s = ncfile["ints"]
                else:
                    ints_s = ncfile.createVariable(
                        "ints",
                        "u1",
                        (
                            "x",
                            "y",
                        ),
                        fill_value=0,
                    )

                logging.debug("Adding to spatial file")
                im_shape = np.array(
                    [
                        int(
                            (
                                global_state.kmer_index.coord_lims[1]
                                - global_state.kmer_index.coord_lims[0]
                            )
                            * render_scale
                        ),
                        int(
                            (
                                global_state.kmer_index.coord_lims[3]
                                - global_state.kmer_index.coord_lims[2]
                            )
                            * render_scale
                        ),
                    ]
                )

                xy = global_state.xy[locs]

                im, _, _ = np.histogram2d(
                    xy[:, 0],
                    xy[:, 1],
                    weights=ints,
                    range=[
                        [
                            global_state.kmer_index.coord_lims[0],
                            global_state.kmer_index.coord_lims[1],
                        ],
                        [
                            global_state.kmer_index.coord_lims[2],
                            global_state.kmer_index.coord_lims[3],
                        ],
                    ],
                    bins=tuple(im_shape),
                )
                im = ((im / im.max()) * 255).astype(np.uint8)

                ints_s[:, :] = im

            self.data = xr.open_dataset(self.outfile_name)
            self.whole_max_ints = []

    def cleanup(self):
        with self.file_lock:
            if self.data is not None:
                self.data.close()
                del self.data
                self.data = None


def get_user_session():
    try:
        if "user_session_id" not in session:
            session["user_session_id"] = str(uuid.uuid4())
            logging.info(f"Created new user_session_id: {session['user_session_id']}")
        return UserSession.get_instance(session["user_session_id"])
    except Exception as e:
        logging.error(f"Error in get_user_session: {str(e)}")
        raise


@app.before_request
def before_request():
    try:
        logging.info(f"Processing request for path: {request.path}")
        logging.debug(f"Request headers: {request.headers}")
        logging.debug(f"Session data: {session}")

        # we do not handle the session creation if we are not accessing /tiles/
        # TODO: investigate why this leads to NetCDF segmentation fault if the code
        # below does not have the if... seems to be the URL for favicons
        g.user_session = get_user_session()
        if request.path.startswith("/tiles/") or (
            request.path == "/" and request.method == "POST"
        ):
            g.user_session.generate_netcdf_index()
    except Exception as e:
        logging.error(f"Error in before_request: {str(e)}")
        return jsonify(error="An internal server error occurred"), 500


# TODO: adapt this so it works with the new coordinate storage method
def interactive_query_standard(
    sequence,
    sliding_size=128,
    pct_threshold=0.65,
    low_complexity_filter=True,
    countmaxkmer=100_000,
    countminkmer=10,
):
    user_session = get_user_session()

    logging.info(f"Querying sequence '{sequence}'")
    locs, ints, where_abundant = global_state.kmer_index.where(
        sequence,
        sliding_size=sliding_size,
        pct_threshold=pct_threshold,
        count_at_most=int(countmaxkmer),
        count_at_least=int(countminkmer),
        use_background_model=False,
    )
    logging.info(f"Adding result to spatial file")
    user_session.add_kmer_to_netcdf_index(locs, ints.astype(np.int32))

    return locs, ints, where_abundant


def interactive_query(
    query,
    sliding_size=128,
    pct_threshold=0.65,
    countmaxkmer=100_000,
    countminkmer=10,
    *args,
    **kwargs,
):
    if query == "":
        raise ValueError("Please specify a query")

    pm, pms = handle_natural_query(query)

    user_session = get_user_session()

    logging.info(f"Querying positive markers for query '{query}', response '{pm}'")
    # locs_p, ints_p, where_abundant = global_state.kmer_index.where(pms, sliding_size=sliding_size, pct_threshold=pct_threshold, query_jump=False, count_at_most=int(countmaxkmer), count_at_least=int(countminkmer))
    locs, ints, where_abundant = global_state.kmer_index.where(
        pms,
        sliding_size=sliding_size,
        pct_threshold=pct_threshold,
        count_at_most=int(countmaxkmer),
        count_at_least=int(countminkmer),
        use_background_model=False,
    )
    # logging.info(f"Querying negative markers '{sequence}'")
    # locs_n, ints_n, where_abundant = global_state.kmer_index.where(nms, sliding_size=sliding_size, pct_threshold=pct_threshold, query_jump=False, count_at_most=int(countmaxkmer), count_at_least=int(countminkmer))

    logging.info(f"Adding result to spatial file")
    user_session.add_kmer_to_netcdf_index(locs, ints.astype(np.int32))

    return pm, pms, where_abundant


def tile_zoomed_coords(xtile, ytile, zoom):
    n = 2.0**zoom
    xtile_zoom = xtile / n * max(global_state.xmax, global_state.ymax)
    ytile_zoom = ytile / n * max(global_state.xmax, global_state.ymax)
    return (xtile_zoom, ytile_zoom)


def generate_tile(zoom, x, y):
    user_session = get_user_session()
    x_sarray = user_session.data["x"]
    y_sarray = user_session.data["y"]

    _sigma = 1 if zoom <= 1 else 1.5

    xleft, yleft = tile_zoomed_coords(int(x), int(y), int(zoom))
    xright, yright = tile_zoomed_coords(int(x) + 1, int(y) + 1, int(zoom))

    xleft_snapped = x_sarray.sel(x=xleft, method="nearest").values
    yleft_snapped = y_sarray.sel(y=yleft, method="nearest").values
    xright_snapped = x_sarray.sel(x=xright, method="nearest").values
    yright_snapped = y_sarray.sel(y=yright, method="nearest").values

    xcondition = f"x >= {xleft_snapped} and x <= {xright_snapped}"
    ycondition = f"y <= {yright_snapped} and y >= {yleft_snapped}"

    thin_value = 1
    if (xright_snapped - xleft_snapped) * (
        yright_snapped - yleft_snapped
    ) >= MAX_DATA_POINTS:
        thin_value = int(
            np.log10(
                (
                    (
                        (xright_snapped - xleft_snapped)
                        * (yright_snapped - yleft_snapped)
                    )
                    / MAX_DATA_POINTS
                )
            )
            + 1
        )
        frame = user_session.data.thin({"x": thin_value, "y": thin_value}).query(
            x=xcondition, y=ycondition
        )
    else:
        frame = user_session.data.query(x=xcondition, y=ycondition)

    csv = ds.Canvas(
        plot_width=256,
        plot_height=256,
        x_range=(xleft, xright),
        y_range=(yleft, yright),
    )

    agg_all = csv.quadmesh(frame, x="x", y="y", agg=ds.mean("all"))
    img_all = np.nan_to_num(agg_all.data)

    if len(user_session.whole_max_ints) < 4:
        user_session.whole_max_ints = [img_all.max()]

    img_all = gaussian(
        img_all / user_session.whole_max_ints[0] * 255,
        sigma=_sigma,
        preserve_range=True,
    )[:, :, np.newaxis]

    if "ints" in frame.variables:
        agg_ch = csv.quadmesh(frame, x="x", y="y", agg=ds.mean("ints"))
        img_ch = np.nan_to_num(agg_ch.data)

        if len(user_session.whole_max_ints) < 4:
            user_session.whole_max_ints.append(img_ch.max())

        img_ch = gaussian(
            img_ch / user_session.whole_max_ints[1] * 255,
            sigma=_sigma,
            preserve_range=True,
        )[:, :, np.newaxis]
        img_all = np.concatenate([img_all, img_ch], axis=2)

    img_all = np.concatenate(
        [
            img_all,
            np.zeros((256, 256, (3 - img_all.shape[-1]))),
            255 * np.ones((256, 256, 1)),
        ],
        axis=2,
    )
    img_all = np.clip(img_all, 0, 255)
    user_session.whole_max_ints += [255] * (4 - len(user_session.whole_max_ints))

    return Image.fromarray(img_all.astype(np.uint8))


@app.route("/")
def index():
    _xmax, _ymax = tile_zoomed_coords(0.5, 0.5, 0)
    return render_template("index.html", xmax=_xmax, ymax=_ymax)


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


@app.route("/parse_queried", methods=["POST"])
def parse_queried():
    if "where_abundant" not in session:
        return json.dumps({"success": False}), 200, {"ContentType": "application/json"}

    scores = [0]
    if len(session["where_abundant"]) > 2:
        data = np.array(session["where_abundant"])
        df = (
            pd.DataFrame({"pos": data[:, 0], "val": data[:, 1]})
            .groupby("pos")
            .mean()
            .rolling(window=24)
            .mean()
        )
        scores = NormalizeData(np.nan_to_num(df["val"].values)).tolist()

    return (
        json.dumps(
            {
                "success": True,
                "query_term": session["query_term"],
                "sequence": session["query_seq"],
                "scores": scores,
            }
        ),
        200,
        {"ContentType": "application/json"},
    )


@map_bp.route("/", methods=["GET", "POST"])
def index():
    help = ""
    try:
        if request.method == "POST":
            query = request.form.get("selectsequence")
            sliding_size = int(request.form.get("sliding_size"))
            pct_threshold = float(request.form.get("pct_threshold"))
            low_complexity_filter = request.form.get(
                "low_complexity_filter"
            ).lower() in ["true", "1", "True"]
            countmaxkmer = 10 ** float(request.form.get("countmaxkmer"))
            countminkmer = 10 ** float(request.form.get("countminkmer"))
            standard_query = True if request.form.get("queryType") in ['1', 1] else False

            if countminkmer >= countmaxkmer:
                raise ValueError(
                    "k-mer 'in at least' must be smaller than 'in at most'"
                )

            if standard_query:
                seq_processed = handle_sequence(query)

                if any(len(s) > SEQ_MAX_LEN for s in seq_processed):
                    help = HINT_SEQUENCE_QUERY
                    raise Exception(
                        f"Cannot have input text/sequences longer than {SEQ_MAX_LEN}."
                    )

                _, _, where_abundant = interactive_query_standard(
                    seq_processed,
                    sliding_size=sliding_size,
                    pct_threshold=pct_threshold,
                    low_complexity_filter=low_complexity_filter,
                    countmaxkmer=countmaxkmer,
                    countminkmer=countminkmer,
                )
                session["query_seq"] = seq_processed
            else:
                pm, pms, where_abundant = interactive_query(
                    query,
                    sliding_size=sliding_size,
                    pct_threshold=pct_threshold,
                    low_complexity_filter=low_complexity_filter,
                    countmaxkmer=countmaxkmer,
                    countminkmer=countminkmer,
                )
                query = query + '\nHere are some relevant genes:\n' + str(pm)
                session["query_seq"] = pms

            session["query_term"] = query
            session["where_abundant"] = where_abundant

            return (
                json.dumps({"success": True}),
                200,
                {"ContentType": "application/json"},
            )
    except Exception as e:
        return (
            json.dumps({"error": str(e), "help": help}),
            500,
            {"ContentType": "application/json"},
        )


@app.route("/tiles/<zoom>/<int:x>/<int:y>.png")
def tile(x, y, zoom):
    try:
        zoom = float(zoom)
        zoom = round(zoom)
        results = generate_tile(zoom, x, y)
        results_bytes = io.BytesIO()
        results.save(results_bytes, "PNG")
        results_bytes.seek(0)
        return send_file(results_bytes, mimetype="image/png")
    except Exception as e:
        return str(e), 400


def _run_serve(args):
    global SEQ_MAX_LEN
    SEQ_MAX_LEN = args.max_len

    try:
        global_state.initialize(args)
    except Exception as e:
        logging.error(f"Error initializing global state: {str(e)}")
        return

    logging.info(f"Setting up web server at {args.address}:{args.port}")
    app.register_blueprint(map_bp, url_prefix="/")
    app.run(debug=True, host=args.address, port=args.port, use_reloader=False)


if __name__ == "__main__":
    from malva.cli import get_serve_parser

    args = get_serve_parser().parse_args()
    _run_serve(args)
