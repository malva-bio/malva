import io
import logging
import numpy as np
import pandas as pd
from flask import Flask, render_template, send_file, request, session, jsonify, Blueprint, g, url_for, make_response
from flask_session import Session
from flask_cors import CORS
from PIL import Image
import uuid
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union
import datashader as ds
from skimage.filters import gaussian
from scipy.spatial import cKDTree
import threading
from pathlib import Path
import tempfile
import re

# for proxy functionality
from urllib.parse import urljoin
from werkzeug.middleware.proxy_fix import ProxyFix

from malva.index import MalvaIndex
from malva.dbutils import handle_sequence
from malva.serve.reportgen import HTMLReportGenerator
# from malva.utils import check_file_exists
# from malva.serve.modeling import handle_natural_query, setup_model
# from malva.serve.templates.strings import HINT_SEQUENCE_QUERY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Errors
class SequenceValidationError(ValueError):
    """Custom error for sequence validation issues"""
    def __init__(self, message: str, help_text: Optional[str] = None):
        super().__init__(message)
        self.help_text = help_text

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
    
    result = global_state.kmer_index.where(
        sequence,
        sliding_size=sliding_size,
        pct_threshold=pct_threshold,
        count_at_most=int(countmaxkmer),
        count_at_least=int(countminkmer),
        use_background_model=False,
        show_coverage=False,
        force_reload=False,
        max_mem="1M"
    )

    locs, ints, where_abundant = result[0]
    
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
                self.kmer_index = MalvaIndex(index_path, verbose=True)
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
    
    def parse_gene_format(query: str) -> str:
        """Parse various gene format inputs into standard format"""
        if query.startswith('gene:') or query.startswith('ensembl:'):
            return query
        
        if ';' in query and not query.startswith('gene:'):
            return f"gene:{query}"
        
        return f"gene:{query};type:cdna"

    def validate_window_size(sequences: Union[List[str], List[List[str]]], window_size: int, kmer_size: int) -> Tuple[int, Optional[str]]:
        """
        Validate window size against sequence lengths and k-mer size.
        Handles both flat lists of sequences and lists of isoform lists.
        
        Args:
            sequences: List of sequences or list of isoform lists
            window_size: Requested window size
            kmer_size: Size of k-mers used in index
            
        Returns:
            Tuple of (validated_size, warning_message)
        """
        if window_size < kmer_size:
            raise SequenceValidationError(
                f"Window size ({window_size}) cannot be smaller than k-mer size ({kmer_size})",
                "Please increase the window size parameter"
            )
        
        # Flatten sequences if needed to find shortest length
        flat_sequences = []
        for seq in sequences:
            if isinstance(seq, list):
                # This is a list of isoforms
                flat_sequences.extend(seq)
            else:
                # This is a single sequence
                flat_sequences.append(seq)
        
        if not flat_sequences:
            raise SequenceValidationError(
                "No valid sequences to process",
                "No sequences were found for the provided input"
            )
        
        min_seq_length = min(len(seq) for seq in flat_sequences)
        
        if min_seq_length < kmer_size:
            raise SequenceValidationError(
                f"Shortest sequence length ({min_seq_length}) cannot be smaller than k-mer size ({kmer_size})",
                "One or more sequences are too short for analysis"
            )
        
        if window_size > min_seq_length:
            new_size = min_seq_length
            return new_size, f"Window size adjusted to {new_size} to match shortest sequence/isoform"
        
        return window_size, None

    def process_sequence_input(query: str) -> List[str]:
        """
        Process input query into list of sequences.
        Handles raw sequences, FASTA format, and gene IDs.
        Supports multiple separators: commas, spaces, newlines
        """
        query = query.strip()
        if not query:
            return []

        # Check if this is a FASTA sequence
        if query.startswith('>'):
            sequences = []
            current_seq = []
            
            for line in query.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('>'):
                    if current_seq:
                        sequences.append('\n'.join(current_seq))
                    current_seq = [line]
                else:
                    current_seq.append(line)
            
            if current_seq:
                sequences.append('\n'.join(current_seq))
                
            return sequences

        # Not FASTA - handle as raw sequence(s) or gene IDs
        parts = []
        for line in query.split('\n'):
            line_parts = re.split(r'[,\s]+', line.strip())
            parts.extend(part for part in line_parts if part)
        
        sequences = []
        for part in parts:
            cleaned = part.strip()
            
            if not cleaned:
                continue
                
            if cleaned.startswith('gene:') or cleaned.startswith('ensembl:'):
                if ';type:' not in cleaned:
                    cleaned = f"{cleaned};type:cdna"
                sequences.append(cleaned)
                
            elif re.match(r'^[ACGTNUacgtnu]+$', cleaned):
                sequences.append(cleaned.upper())

            else:
                sequences.append(f"gene:{cleaned};type:cdna")

        return sequences

    @app.route("/search", methods=["POST"])
    def search():
        try:
            query = request.args.get("selectsequence", "").strip()
            if not query:
                raise SequenceValidationError("No sequence provided", 
                    "Please enter a sequence or gene name")
            
            logger.info("Starting sequence processing...")
            
            # Get parameters
            sliding_size = int(request.args.get("sliding_size", 128))
            pct_threshold = float(request.args.get("pct_threshold", 0.65))
            low_complexity_filter = request.args.get("low_complexity_filter", "").lower() in ["true", "1"]
            countmaxkmer = int(float(request.args.get("countmaxkmer", 5)))
            countminkmer = int(float(request.args.get("countminkmer", 1)))
            
            # Process input
            processed_sequences = process_sequence_input(query)
            logger.info(f"Initial processing returned {len(processed_sequences)} sequences")
            
            if not processed_sequences:
                raise SequenceValidationError(
                    "No valid sequences found in input",
                    "Check the format of your input. Examples are shown below."
                )
            
            # Handle gene IDs and get actual sequences
            final_sequences = []
            warnings = []
            errors = []
            
            for seq in processed_sequences:
                try:
                    if seq.startswith('>'):  # FASTA
                        lines = seq.split('\n')
                        sequence = ''.join(lines[1:]).upper()
                        if not re.match(r'^[ACGTNU]+$', sequence):
                            errors.append(f"Invalid FASTA sequence: {lines[0]}")
                            continue
                        final_sequences.append(sequence)
                        
                    elif seq.startswith('gene:') or seq.startswith('ensembl:') or ';' in seq:
                        # Handle gene IDs
                        result = handle_sequence(seq)
                        if result is None:
                            errors.append(f"Gene not found: {seq}")
                            continue
                            
                        if isinstance(result, list):
                            final_sequences.extend(result)
                        else:
                            final_sequences.append(result)
                            
                    else:
                        # Must be raw sequence or simple gene
                        if re.match(r'^[ACGTNUacgtnu]+$', seq):
                            # It's a raw sequence
                            final_sequences.append(seq.upper())
                        else:
                            # Try as simple gene name
                            result = handle_sequence(f"gene:{seq}")
                            if result is None:
                                errors.append(f"Gene not found: {seq}")
                                continue
                                
                            if isinstance(result, list):
                                final_sequences.extend(result)
                            else:
                                final_sequences.append(result)
                                
                except Exception as e:
                    errors.append(f"Error processing {seq[:20]}: {str(e)}")
                    continue
            
            logger.info(f"Final processing yielded {len(final_sequences)} sequences")
            
            if not final_sequences:
                error_msg = "No valid sequences could be processed."
                if errors:
                    error_msg += f" Errors: {'; '.join(errors)}"
                raise SequenceValidationError(error_msg, 
                    "Please check your input and try again")
            
            # Validate window size
            kmer_size = global_state.kmer_index.kmer_size
            min_seq_length = min(len(seq) for seq in final_sequences)
            
            if min_seq_length < kmer_size:
                raise SequenceValidationError(
                    f"Sequence length ({min_seq_length}) cannot be smaller than k-mer size ({kmer_size})",
                    "One or more sequences are too short for analysis"
                )
            
            if sliding_size > min_seq_length:
                sliding_size = min_seq_length
                warnings.append(f"Window size adjusted to {sliding_size} to match shortest sequence")
            
            # Query the index
            try:
                locs, ints, where_abundant = interactive_query_standard(
                    final_sequences,
                    sliding_size=sliding_size,
                    pct_threshold=pct_threshold,
                    low_complexity_filter=low_complexity_filter,
                    countmaxkmer=10**countmaxkmer,
                    countminkmer=10**countminkmer
                )
            except Exception as e:
                logger.error(f"Query error: {str(e)}")
                raise SequenceValidationError(
                    "Error searching sequences",
                    "Try adjusting the search parameters"
                )
            
            # Check results
            if len(locs) == 0:
                raise SequenceValidationError(
                    "No results found for any sequences",
                    "Try adjusting the search parameters"
                )
            
            # Store results
            g.user_session.add_query_result(
                global_state.xy[locs],
                ints
            )
            
            # Store first sequence for display
            session["query_seq"] = final_sequences[0]
            session["query_term"] = query
            session["where_abundant"] = where_abundant.tolist() if isinstance(where_abundant, np.ndarray) else where_abundant
            
            response = {"success": True}
            if warnings:
                response["warnings"] = warnings
            if errors:
                response["errors"] = errors
            
            logger.info("Search completed successfully")
            return jsonify(response)
            
        except SequenceValidationError as e:
            return jsonify({
                "error": str(e),
                "help": e.help_text
            }), 400
        except Exception as e:
            logger.error(f"Unexpected error in search: {str(e)}")
            return jsonify({
                "error": "An unexpected error occurred",
                "help": "Please try again or contact support if the problem persists"
            }), 500
        
    # Add route to handle save request
    @app.route("/save_report", methods=["POST"])
    def save_report():
        try:
            # Create report
            generator = HTMLReportGenerator(session)
            zip_data = generator.create_report_zip()
            
            # Create response
            response = make_response(zip_data)
            response.headers['Content-Type'] = 'application/zip'
            response.headers['Content-Disposition'] = 'attachment; filename=malva_report.zip'
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return jsonify({
                "error": "Failed to generate report",
                "help": "Please try again or contact support if the problem persists"
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