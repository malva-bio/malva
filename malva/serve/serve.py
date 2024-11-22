import io
import logging
import numpy as np
import pandas as pd
from flask import Flask, render_template, send_file, request, session, jsonify, Blueprint, g, url_for
from flask_session import Session
from flask_cors import CORS
from PIL import Image
import uuid
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List
import datashader as ds
from skimage.filters import gaussian
from scipy.spatial import cKDTree
import threading
from pathlib import Path
import tempfile

# for proxy functionality
from urllib.parse import urljoin
from werkzeug.middleware.proxy_fix import ProxyFix

from malva.index import MalvaIndex
from malva.dbutils import handle_sequence
# from malva.utils import check_file_exists
# from malva.serve.modeling import handle_natural_query, setup_model
# from malva.serve.templates.strings import HINT_SEQUENCE_QUERY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_LOAD_ALL = 10_000_000
TILE_SIZE = 256
MAX_SEQUENCE_LENGTH = 1000

EMPTY_TILE = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
EMPTY_TILE_BYTES = io.BytesIO()
Image.fromarray(EMPTY_TILE).save(EMPTY_TILE_BYTES, format='PNG')
EMPTY_TILE_BYTES.seek(0)

def interactive_query_standard(
    sequence: str,
    sliding_size: int = 128,
    pct_threshold: float = 0.65,
    low_complexity_filter: bool = True,
    countmaxkmer: int = 100_000,
    countminkmer: int = 10,
) -> Tuple[np.ndarray, np.ndarray, List]:
    """Process standard sequence query"""
    logger.info(f"Querying sequence '{sequence}'")
    
    locs, ints, where_abundant = global_state.kmer_index.where(
        sequence,
        sliding_size=sliding_size,
        pct_threshold=pct_threshold,
        count_at_most=int(countmaxkmer),
        count_at_least=int(countminkmer),
        use_background_model=False,
    )
    
    return locs, ints, where_abundant

class MalvaProxyFix:
    """Custom middleware for handling proxy prefixes"""
    def __init__(self, app, uuid):
        self.app = app
        self.prefix = f"/api/malva/view/{uuid}"
        
    def __call__(self, environ, start_response):
        # Ensure PATH_INFO contains the full path
        if environ['PATH_INFO'].startswith('/view/'):
            # Reconstruct the full path if it's not present
            environ['PATH_INFO'] = f"/api/malva{environ['PATH_INFO']}"
        
        # Now handle the complete path
        if environ['PATH_INFO'].startswith(self.prefix):
            script_name = self.prefix
            path_info = environ['PATH_INFO'][len(self.prefix):]
            
            # Ensure path_info starts with a slash
            if path_info and not path_info.startswith('/'):
                path_info = '/' + path_info
                
            environ['SCRIPT_NAME'] = script_name
            environ['PATH_INFO'] = path_info
            
        return self.app(environ, start_response)

@dataclass
class SpatialData:
    """Class to hold spatial data and related methods"""
    coordinates: np.ndarray  # Nx2 array of x,y coordinates
    values: np.ndarray      # N-length array of values
    bounds: Tuple[float, float, float, float]  # xmin, xmax, ymin, ymax
    
    @property
    def xmin(self) -> float:
        return self.bounds[0]
    
    @property
    def xmax(self) -> float:
        return self.bounds[1]
    
    @property
    def ymin(self) -> float:
        return self.bounds[2]
    
    @property
    def ymax(self) -> float:
        return self.bounds[3]

class GlobalState:
    """Singleton class to manage global state and data"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.coordinates = None
            self.values = None
            self.bounds = None
            self.initialized = False

    def initialize_from_index(self, index_path: str, max_mem: float = 32):
        """Initialize global state from a malva index file"""
        with self._lock:
            try:
                logger.info("Loading malva index and metadata")
                self.kmer_index = MalvaIndex(index_path)
                self.kmer_index.open()

                # Load the initial data points
                if len(self.kmer_index.index[f"index_0_data"]) >= 2*MAX_LOAD_ALL:
                    self._loc_all, self._abu_all = np.unique(
                        self.kmer_index.index[f"index_0_data"][MAX_LOAD_ALL:(2*MAX_LOAD_ALL)],
                        return_counts=True,
                    )
                else:
                    self._loc_all, self._abu_all = np.unique(
                        self.kmer_index.index[f"index_0_data"][0:MAX_LOAD_ALL],
                        return_counts=True,
                    )

                logger.info("Loading the pointers into memory. Might take a while.")
                self.kmer_index.where(
                    "A" * self.kmer_index.kmer_size + "T",
                    max_mem=max_mem,
                    use_background_model=False
                )

                # Set up the coordinate bounds
                self.bounds = (
                    self.kmer_index.coord_lims[0],  # xmin
                    self.kmer_index.coord_lims[1] + 1,  # xmax
                    self.kmer_index.coord_lims[2],  # ymin
                    self.kmer_index.coord_lims[3] + 1   # ymax
                )
                
                # Store the spatial coordinates
                self.xy = self.kmer_index.spatial_coord[:]
                self.initialized = True
                
            except Exception as e:
                logger.error(f"Error initializing from index: {str(e)}")
                raise
    
    def initialize(self, coordinates: np.ndarray, values: np.ndarray, bounds: Tuple):
        """Initialize with data"""
        with self._lock:
            self.coordinates = coordinates
            self.values = values
            self.bounds = bounds
            self.initialized = True

    def get_bounds(self):
        """Get the coordinate bounds"""
        if not self.initialized:
            raise RuntimeError("GlobalState not initialized")
        return self.bounds

    def get_coordinates(self):
        """Get the spatial coordinates"""
        if not self.initialized:
            raise RuntimeError("GlobalState not initialized")
        return self.xy if self.xy is not None else self.coordinates

class UserSession:
    """Class to manage per-user session data"""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.data_path = Path(tempfile.gettempdir()) / f"user_{session_id}.nc"
        self.lock = threading.Lock()
        self.data = None
        self.query_results = None
        self.max_values = []
        self.background_data = None
        # Store global intensity ranges for consistent scaling
        self.background_intensity_range = None
        self.query_intensity_range = None
        # Store trees for fast spatial lookup
        self.background_tree = None
        self.query_tree = None
    
    def initialize_background(self):
        """Initialize session with background data from global state"""
        if global_state.initialized and global_state._loc_all is not None:
            with self.lock:
                logger.info("Initializing background data for session")
                background_coords = global_state.xy[global_state._loc_all]
                background_values = global_state._abu_all
                
                # Calculate global intensity range for background
                self.background_intensity_range = self._calculate_intensity_range(background_values)
                
                self.background_data = {
                    'locations': background_coords,
                    'intensities': background_values
                }
                self.background_tree = cKDTree(background_coords)
                logger.info(f"Background data initialized with intensity range: {self.background_intensity_range}")

    def _calculate_intensity_range(self, values: np.ndarray) -> Tuple[float, float]:
        """Calculate robust intensity range for values"""
        if len(values) == 0:
            return (0, 1)
        
        # Use percentiles for more robust range estimation
        min_val = np.percentile(values, 1)  # 1st percentile instead of min
        max_val = np.percentile(values, 99)  # 99th percentile instead of max
        
        # Ensure we don't have zero range
        if min_val == max_val:
            max_val = min_val + 1
            
        return (min_val, max_val)

    def add_query_result(self, locations: np.ndarray, intensities: np.ndarray):
        """Add query results to user session"""
        with self.lock:
            # Calculate global intensity range for query
            self.query_intensity_range = self._calculate_intensity_range(intensities)
            
            self.query_results = {
                'locations': locations,
                'intensities': intensities
            }
            self.query_tree = cKDTree(locations)
            logger.info(f"Query results added with intensity range: {self.query_intensity_range}")


    def add_background_data(self, coordinates: np.ndarray, values: np.ndarray):
        """Store background data"""
        with self.lock:
            self.background_data = {
                'locations': coordinates,
                'intensities': values
            }
            self.background_tree = cKDTree(coordinates)

    def _rasterize_points(self, points: np.ndarray, values: np.ndarray, tree: cKDTree,
                     x_range: Tuple[float, float], y_range: Tuple[float, float],
                     value_range: Optional[Tuple[float, float]] = None,
                     sigma: float = 1.0) -> np.ndarray:
        """Convert points to raster image preserving aspect ratio"""
        if points is None or len(points) == 0:
            return np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
        
        # Find the largest range to maintain aspect ratio
        x_size = x_range[1] - x_range[0]
        y_size = y_range[1] - y_range[0]
        max_range = max(x_size, y_size)
        half_range = max_range * 0.5
        
        x_center = (x_range[0] + x_range[1]) * 0.5
        y_center = (y_range[0] + y_range[1]) * 0.5

        # Query point for the radius search
        query_point = np.array([x_center, y_center])
        radius = np.sqrt(2) * half_range
        
        x_adjusted = (x_center - half_range, x_center + half_range)
        y_adjusted = (y_center - half_range, y_center + half_range)

        # Query points using spatial index
        indices = tree.query_ball_point(query_point, radius)

        if not indices:
            return np.zeros(TILE_SIZE * TILE_SIZE, dtype=np.uint8).reshape(TILE_SIZE, TILE_SIZE)

        # Use queried points
        filtered_points = points[indices]
        filtered_values = values[indices]

        # Additional filter for exact box bounds
        x_adjusted = (x_center - half_range, x_center + half_range)
        y_adjusted = (y_center - half_range, y_center + half_range)
        mask = ((filtered_points[:, 0] >= x_adjusted[0]) & 
                (filtered_points[:, 0] < x_adjusted[1]) & 
                (filtered_points[:, 1] >= y_adjusted[0]) & 
                (filtered_points[:, 1] < y_adjusted[1]))
        
        # Use datashader with adjusted ranges
        canvas = ds.Canvas(plot_width=TILE_SIZE,
                         plot_height=TILE_SIZE,
                         x_range=x_adjusted,
                         y_range=y_adjusted)
        
        df = pd.DataFrame({
            'x': filtered_points[:, 0],
            'y': filtered_points[:, 1],
            'val': filtered_values
        })
        
        agg = canvas.points(df, 'x', 'y', agg=ds.mean('val'))
        img = np.nan_to_num(agg.values, copy=False)
        
        # Use global value range for normalization if provided
        if value_range is not None and value_range[1] > value_range[0]:
            img = np.clip(img, value_range[0], value_range[1])
            img = (img - value_range[0]) / (value_range[1] - value_range[0]) * 255
        else:
            if img.max() > 0:
                img = img / img.max() * 255
        
        img = gaussian(img, sigma=sigma, preserve_range=True)
        return img.astype(np.uint8)
        
    def get_tile_data(self, x: int, y: int, zoom: int) -> np.ndarray:
        """Get tile data for this user's view"""
        if not global_state.initialized:
            raise RuntimeError("Global state not initialized")
        
        # Get global coordinate limits
        coord_lims = global_state.kmer_index.coord_lims
        x_total = coord_lims[1] - coord_lims[0]
        y_total = coord_lims[3] - coord_lims[2]
        max_dim = max(x_total, y_total)
        
        # Calculate tile bounds based on the maximum dimension
        n = 2.0 ** zoom
        tile_size = max_dim / n
        
        x_bounds = (
            coord_lims[0] + x * tile_size,
            coord_lims[0] + (x + 1) * tile_size
        )
        y_bounds = (
            coord_lims[2] + y * tile_size,
            coord_lims[2] + (y + 1) * tile_size
        )
        
        _sigma = 1 if zoom <= 1 else 1.2

        # Initialize empty RGBA image
        img_all = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)

        # Add background data to first channel (red)
        if hasattr(self, 'background_data') and self.background_data is not None:
            img_all[:, :, 0] = self._rasterize_points(
                self.background_data['locations'],
                self.background_data['intensities'],
                self.background_tree,
                x_bounds, 
                y_bounds,
                value_range=self.background_intensity_range,
                sigma=_sigma
            )

        # Add query results to second channel (green)
        if self.query_results is not None and len(self.query_results.get('locations', [])) > 0:
            img_all[:, :, 1] = self._rasterize_points(
                self.query_results['locations'],
                self.query_results['intensities'],
                self.query_tree,
                x_bounds,
                y_bounds,
                value_range=self.query_intensity_range,
                sigma=_sigma
            )

        # Set alpha channel only where we have data
        img_all[:, :, 3] = np.maximum(img_all[:, :, 0], img_all[:, :, 1]) > 0
        img_all[:, :, 3] *= 255

        return img_all
        
    def cleanup(self):
        """Clean up session data"""
        with self.lock:
            if self.data_path.exists():
                self.data_path.unlink()

def create_app(init_state=True, _uuid=None):
    """Application factory function"""
    app = Flask(__name__)

    # Configure app
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev_key"),
        SESSION_TYPE="filesystem",
        SESSION_PERMANENT=True,
    )

    if _uuid:
        # Set application root for proper URL generation
        app.config['APPLICATION_ROOT'] = f"/api/malva/view/{_uuid}/"
        
        # Apply proxy fixes in correct order
        app.wsgi_app = MalvaProxyFix(app.wsgi_app, _uuid)
        app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

    Session(app)
    
    # Session management
    user_sessions = {}
    session_lock = threading.Lock()
    
    def get_user_session():
        """Get or create user session"""
        if 'user_id' not in session:
            session['user_id'] = str(uuid.uuid4())
        
        user_id = session['user_id']
        with session_lock:
            if user_id not in user_sessions:
                new_session = UserSession(user_id)
                if init_state and global_state.initialized:
                    new_session.initialize_background()
                user_sessions[user_id] = new_session
            return user_sessions[user_id]
    
    @app.before_request
    def before_request():
        """Set up user session before each request
        
        Note: we should never get these via the /api/malva/view
        or other
        """
        if not request.path.startswith("/health"):
            try:
                g.user_session = get_user_session()
            except Exception as e:
                logger.error(f"Error in before_request: {str(e)}")
                return jsonify({"error": "Internal server error"}), 500

    
    @app.route("/parse_queried", methods=["POST"])
    def parse_queried():
        """Process and return query results"""
        try:
            if "where_abundant" not in session:
                return jsonify({
                    "success": False,
                    "query_term": "",
                    "sequence": "",
                    "scores": []
                })

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
                # Convert numpy values to native Python types for proper JSON serialization
                scores = df["val"].fillna(0).tolist()
                if scores:  # Normalize only if we have scores
                    min_val = min(scores)
                    max_val = max(scores)
                    if max_val > min_val:
                        scores = [(x - min_val) / (max_val - min_val) for x in scores]

            response_data = {
                "success": True,
                "query_term": session.get("query_term", ""),
                "sequence": session.get("query_seq", ""),
                "scores": scores
            }
            
            # Set correct content type header
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"Error in parse_queried: {str(e)}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @app.route("/tiles/<zoom>/<int:x>/<int:y>.png")
    def tile(zoom: str, x: int, y: int):
        """Generate and serve tile image"""
        try:
            zoom = int(float(zoom))

            coord_lims = global_state.kmer_index.coord_lims
            
            # Calculate tile bounds
            n = 2.0 ** zoom
            x_tile_min = x / n * (coord_lims[1] - coord_lims[0]) + coord_lims[0]
            x_tile_max = (x + 1) / n * (coord_lims[1] - coord_lims[0]) + coord_lims[0]
            y_tile_min = y / n * (coord_lims[3] - coord_lims[2]) + coord_lims[2]
            y_tile_max = (y + 1) / n * (coord_lims[3] - coord_lims[2]) + coord_lims[2]
            
            # Check if tile is completely outside data bounds
            if (x_tile_min > coord_lims[1] or 
                x_tile_max < coord_lims[0] or
                y_tile_min > coord_lims[3] or
                y_tile_max < coord_lims[2]):
                # Return pre-generated transparent tile
                return send_file(
                    EMPTY_TILE_BYTES, 
                    mimetype='image/png',
                )

            img_array = g.user_session.get_tile_data(x, y, zoom)
            
            # Convert numpy array to PIL Image and serve
            img_io = io.BytesIO()
            Image.fromarray(img_array).save(
                img_io, 
                format='PNG',
                optimize=True,
                compress_level=1  # Faster compression
            )
            img_io.seek(0)
            
            return send_file(img_io, mimetype='image/png')
        except Exception as e:
            logger.error(f"Error generating tile: {str(e)}")
            return str(e), 400
    
    @app.route("/health")
    def health():
        """Health check endpoint"""
        return jsonify({
            "status": "healthy" if global_state.initialized else "initializing",
            "message": "Service is ready" if global_state.initialized else "Service is initializing"
        })

    @app.route("/search", methods=["POST"])
    def search():
        try:
            # Get parameters from URL args instead of form
            query = request.args.get("selectsequence", "")
            sliding_size = int(request.args.get("sliding_size", 128))
            pct_threshold = float(request.args.get("pct_threshold", 0.65))
            low_complexity_filter = request.args.get("low_complexity_filter", "").lower() in ["true", "1", "true"]
            countmaxkmer = 10 ** float(request.args.get("countmaxkmer", "5"))
            countminkmer = 10 ** float(request.args.get("countminkmer", "1"))
            standard_query = request.args.get("queryType") in ['1', 1]

            # Print received parameters for debugging
            print("Received parameters:", {
                "query": query,
                "sliding_size": sliding_size,
                "pct_threshold": pct_threshold,
                "low_complexity_filter": low_complexity_filter,
                "countmaxkmer": countmaxkmer,
                "countminkmer": countminkmer,
                "standard_query": standard_query
            })

            if countminkmer >= countmaxkmer:
                raise ValueError("k-mer 'in at least' must be smaller than 'in at most'")

            seq_processed = handle_sequence(query)

            if len(seq_processed) > MAX_SEQUENCE_LENGTH:
                raise ValueError(f"Cannot have input sequences longer than {MAX_SEQUENCE_LENGTH}")

            locs, ints, where_abundant = interactive_query_standard(
                seq_processed,
                sliding_size=sliding_size,
                pct_threshold=pct_threshold,
                low_complexity_filter=low_complexity_filter,
                countmaxkmer=countmaxkmer,
                countminkmer=countminkmer,
            )

            print(f"Query results: {len(locs)} locations, {len(ints)} intensities")
            
            # Store results in user session
            g.user_session.add_query_result(
                global_state.xy[locs],  # Convert indices to coordinates
                ints
            )
            
            session["query_seq"] = seq_processed
            session["query_term"] = query
            session["where_abundant"] = where_abundant
            
            return jsonify({"success": True})
        except Exception as e:
            print("Error processing query:", str(e))
            return jsonify({
                "error": str(e),
                "help": "Please check your input parameters and try again"
            }), 500

    @app.route("/", methods=["GET"])
    def index():
        """Handle root path requests"""
        try:
            _xmax = global_state.kmer_index.coord_lims[1] + 1
            _ymax = global_state.kmer_index.coord_lims[3] + 1
            return render_template("index.html", xmax=_xmax, ymax=_ymax)
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return app

def _run_serve(args):
    """Run the server with the given arguments"""
    global MAX_SEQUENCE_LENGTH, global_state
    MAX_SEQUENCE_LENGTH = args.max_len
    
    try:
        # Initialize global state first
        global_state = GlobalState()
        global_state.initialize_from_index(args.index_in, max_mem=args.max_mem)
        logger.info("Initialized global state from index")
        
        # Create app only once after global state is initialized
        app = create_app(init_state=True, _uuid=args.uuid)

        CORS(app, resources={
            r"/*": {
                "origins": "*",
                "allow_headers": ["Content-Type", "Authorization"],
                "supports_credentials": True
            }
        })

        app.run(
            debug=False, 
            host=args.address, 
            port=args.port, 
            use_reloader=False
        )
    except Exception as e:
        logger.error(f"Error running serve: {str(e)}")
        raise

app = None

if __name__ == "__main__":
    from malva.cli import get_serve_parser

    args = get_serve_parser().parse_args()
    _run_serve(args)