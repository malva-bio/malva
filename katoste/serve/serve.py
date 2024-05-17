import io
import logging
import numpy as np
import json

from netCDF4 import Dataset
import xarray as xr
import datashader as ds

from flask import Flask, render_template, send_file, Blueprint, request
from skimage.filters import gaussian
from PIL import Image

from scipy.interpolate import splrep, BSpline

from katoste.index import KatosteIndex
from katoste.utils import check_file_exists
from katoste.dbutils import handle_sequence

SEQ_MAX_LEN = 1_000

app = Flask(__name__)
map_bp = Blueprint('map', __name__)

def interactive_query(sequence):
    global kmer_index, xy, x_sarray, y_sarray, where_abundant

    logging.info(f"Querying sequence '{sequence}'")
    locs, ints, where_abundant = kmer_index.where(sequence)
    add_kmer_to_netcdf_index(xy[locs, 0], xy[locs, 1], ints.astype(np.int32))

    return locs, ints

def _run_serve(args):
    global app, kmer_index, xy, xmax, ymax, _loc_all, _abu_all, data, whole_max_ints, where_abundant, queried, query_seq, query_term
    whole_max_ints = []

    logging.info("Loading katoste index and metadata")
    kmer_index = KatosteIndex(args.index_in)
    kmer_index.open() # TODO: when loading, if the index exists, instead of this

    # _all_kmers = kmer_index._index_backed['index_0_indptr']
    # _kmer_abundance = np.diff(np.concatenate([_all_kmers, np.array(kmer_index.index[f'index_0_data'].shape[0])]))
    # _kmer_filtered = _kmer_abundance
    _loc_all, _abu_all = np.unique(kmer_index.index[f'index_0_data'][:], return_counts=True)
    # _ix_ab = np.argsort(_abu_all)[::-1][100:]

    # _loc_all = _loc_all[_ix_ab]
    # _abu_all = _abu_all[_ix_ab]

    kmer_index.close()

    logging.info("Loading the pointers into memory. Might take a while.")
    kmer_index.where("A"*kmer_index.kmer_size, lazy_index=args.lazy_index)
    where_abundant = []
    query_seq = ""
    query_term = ""
    queried = False

    xmax = kmer_index.coord_lims[1]+1
    ymax = kmer_index.coord_lims[3]+1
    n_spatial = kmer_index.n_spatial

    xy = np.unravel_index(np.arange(n_spatial), (xmax, ymax), order='C')
    xy = np.vstack(xy).T

    logging.info(f"Create temporary spatial plotter index as netCDF")    
    generate_netcdf_index(kmer_index, xy, xmax, ymax)
    data = xr.open_dataset("test.nc")
    generate_tile(0, 0, 0) # run once to preload

    logging.info(f"Setting up web server at {args.address}:{args.port}")
    app.register_blueprint(map_bp, url_prefix="/")
    app.run(debug=True, host=args.address, port=args.port, use_reloader=False)

def generate_netcdf_index(kmer_index, xy, xmax, ymax):
    ncfile = Dataset("test.nc", "w", format="NETCDF4")

    x_dim = ncfile.createDimension('x', kmer_index.coord_lims[1]+1)
    y_dim = ncfile.createDimension('y', kmer_index.coord_lims[3]+1)
    nc_all = ncfile.createDimension("all", None)

    x = ncfile.createVariable('x', np.uint32, ('x',))
    y = ncfile.createVariable('y', np.uint32, ('y',))
    
    x[:] = np.arange(kmer_index.coord_lims[1]+1)
    y[:] = np.arange(kmer_index.coord_lims[3]+1)

    all_s = ncfile.createVariable("all",'u1',("x","y",), fill_value=0)

    _z = np.zeros((xmax, ymax))
    _z[xy[_loc_all, 0], xy[_loc_all, 1]] = _abu_all
    all_s[:, :] = _z

    ncfile.close()

def tile_zoomed_coords(xtile, ytile, zoom):
    n = 2.0 ** zoom
    xtile_zoom = xtile / n * max(xmax, ymax)
    ytile_zoom = ytile / n * max(xmax, ymax)

    return (xtile_zoom, ytile_zoom)

def generate_tile(zoom, x, y):
    """
    The function takes the zoom and tile path from the web request,
    and determines the top left and bottom right coordinates of the tile.
    This information is used to query against the dataframe.
    """
    global data, xmax, ymax, whole_max_ints

    x_sarray = data['x']
    y_sarray = data['y']

    _sigma = 0.5 if zoom <= 1 else 1.5
    
    xleft, yleft = tile_zoomed_coords(int(x), int(y), int(zoom))
    xright, yright = tile_zoomed_coords(int(x)+1, int(y)+1, int(zoom))

    # ensures no gaps are left between tiles due to partitioning
    xleft_snapped = x_sarray.sel(x=xleft, method="nearest").values  
    yleft_snapped = y_sarray.sel(y=yleft, method="nearest").values
    xright_snapped = x_sarray.sel(x=xright, method="nearest").values
    yright_snapped = y_sarray.sel(y=yright, method="nearest").values

    # ... and needs to be >= and <= so there are no gaps
    xcondition = f"x >= {xleft_snapped} and x <= {xright_snapped}"
    ycondition = f"y <= {yright_snapped} and y >= {yleft_snapped}"
    frame = data.query(x=xcondition, y=ycondition)
    #frame = frame.coarsen(x=(6-zoom), y=(6-zoom), boundary='trim').mean()

    csv = ds.Canvas(plot_width=256, plot_height=256,
                    x_range=(xleft, xright), y_range=(yleft, yright))
    
    agg_all = csv.quadmesh(frame, x='x', y='y', agg=ds.mean('all'))
    img_all = np.nan_to_num(agg_all.data)

    if len(whole_max_ints) < 4:
        whole_max_ints = [img_all.max()]

    img_all = gaussian(img_all / whole_max_ints[0] * 255, sigma=_sigma, preserve_range=True)[:, :, np.newaxis]

    if 'ints' in frame.variables:
        agg_ch = csv.quadmesh(frame, x='x', y='y', agg=ds.mean('ints'))
        img_ch = np.nan_to_num(agg_ch.data)
        
        if len(whole_max_ints) < 4:
            whole_max_ints.append(img_ch.max())

        img_ch = gaussian(img_ch / whole_max_ints[1] * 255, sigma=_sigma, preserve_range=True)[:, :, np.newaxis]
        img_all = np.concatenate([img_all, img_ch], axis=2)
    
    img_all = np.concatenate([img_all, np.zeros((256, 256, (3 - img_all.shape[-1]))), 255*np.ones((256, 256, 1))], axis=2)
    img_all = np.clip(img_all, 0, 255)
    whole_max_ints += [255]*(4 - len(whole_max_ints))

    return Image.fromarray(img_all.astype(np.uint8))

@app.route("/")
def index():
    _xmax, _ymax = tile_zoomed_coords(0.5, 0.5, 0)
    return render_template("index.html", xmax=_xmax, ymax=_ymax)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

@app.route("/parse_queried", methods=['POST'])
def parse_queried():
    global queried, where_abundant, query_seq, query_term

    if not queried:
        return json.dumps({'success':False}), 200, {'ContentType':'application/json'}
    
    y_spline = [0]
    if len(where_abundant) > 2:
        _where_abundant = np.array(where_abundant)
        x = _where_abundant[:, 0]
        y = _where_abundant[:, 1]
        x_new = np.arange(0, max(x))

        tck = splrep(x, y, s=100_000, k=1)
        y_spline = BSpline(*tck)(x_new)
        y_spline = NormalizeData(y_spline).tolist()
    
    return json.dumps({'success':True, 'query_term': query_term, 'sequence': query_seq, 'scores': y_spline}), 200, {'ContentType':'application/json'} 

@map_bp.route("/", methods=['GET', 'POST'])
def index():
    global queried, query_seq, query_term
    try:
        if request.method == 'POST':
            seq = request.form.get("selectsequence")
            seq_processed = handle_sequence(seq)
            if len(seq_processed) > SEQ_MAX_LEN:
                raise Exception(f"Cannot have input text/sequences longer than {SEQ_MAX_LEN}")

            query_seq = seq_processed
            query_term = seq
            interactive_query(seq_processed)
            queried = True
            return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 
    except Exception as e:
        return json.dumps({'error': f'An error occurred {str(e)}'}), 500, {'ContentType':'application/json'} 
        

@app.route("/tiles/<int:zoom>/<int:x>/<int:y>.png")
def tile(x, y, zoom):
    results = generate_tile(zoom, x, y)
    # image passed off to bytestream
    results_bytes = io.BytesIO()
    results.save(results_bytes, 'PNG')
    results_bytes.seek(0)
    return send_file(results_bytes, mimetype='image/png')

def add_kmer_to_netcdf_index(xloc, yloc, ints):
    global data, xmax, ymax, whole_max_ints

    data.close()
    _exists = check_file_exists("test.nc")
    _mode = "w"
    if _exists:
        _mode = "r+"
    
    ncfile = Dataset("test.nc", _mode, format="NETCDF4")

    if _exists and 'ints' in ncfile.variables:
        ints_s = ncfile['ints']
    else:
        ints_s = ncfile.createVariable("ints",'u1',("x","y",), fill_value=0)

    _z = np.ma.zeros((xmax, ymax))
    _z[xloc, yloc] = ints
    ints_s[:, :] = _z

    ncfile.close()
    data = xr.open_dataset("test.nc")
    whole_max_ints = []
    generate_tile(0, 0, 0) # run once to preload
    
if __name__ == "__main__":
    from katoste.cli import get_serve_parser
    args = get_serve_parser().parse_args()
    _run_serve(args)