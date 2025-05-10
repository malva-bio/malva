import numpy as np
import blosc
import os
import pickle
import math
from typing import List, Union, Tuple, Dict
from tqdm import tqdm
import mmap
from pyfastpfor import getCodec, delta1, prefixSum1

blosc.set_nthreads(8)

class LazyChunkTable:
    def __init__(self, filename: str, chunk_table_pos: int, num_chunks: int, buffer_size=1):
        self.filename = filename
        self.chunk_table_pos = chunk_table_pos
        self.num_chunks = num_chunks
        self.entry_size = 24
        self.cache = {}
        self.cache_max_size = 10_000_000
        self.buffer_size = buffer_size
        
        # Keep a single file handle open
        self.file = open(self.filename, 'rb')
        # Use a memory-mapped buffer for faster access
        self.mmap_data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Pre-calculate offset ranges to avoid repeated calculation
        self.chunk_table_end = self.chunk_table_pos + (self.num_chunks * self.entry_size)

    def __len__(self) -> int:
        """Return the number of chunks."""
        return self.num_chunks

    def __getitem__(self, idx):
        """
        Get chunk entry at the specified index.
        Supports integer indexing.
        """
        if isinstance(idx, int):
            return self._load_entry(idx)
        elif isinstance(idx, slice):
            # If someone slices, load the requested range
            start = 0 if idx.start is None else idx.start
            stop = self.num_chunks if idx.stop is None else idx.stop
            step = 1 if idx.step is None else idx.step
            
            return [self._load_entry(i) for i in range(start, min(stop, self.num_chunks), step)]
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")
    
    def __del__(self):
        # Clean up resources
        if hasattr(self, 'mmap_data') and self.mmap_data:
            self.mmap_data.close()
        if hasattr(self, 'file') and self.file:
            self.file.close()
    
    def _load_entries_batch(self, start_chunk_id: int) -> None:
        """Load a batch of entries at once to minimize I/O operations"""
        if start_chunk_id >= self.num_chunks:
            return
            
        end_chunk_id = min(start_chunk_id + self.buffer_size, self.num_chunks)
        start_pos = self.chunk_table_pos + (start_chunk_id * self.entry_size)
        batch_size = (end_chunk_id - start_chunk_id) * self.entry_size
        
        # Read the entire batch at once from mmap
        batch_data = self.mmap_data[start_pos:start_pos + batch_size]
        
        # Parse entries and update cache
        for i in range(start_chunk_id, end_chunk_id):
            offset = (i - start_chunk_id) * self.entry_size
            chunk_id_read = int.from_bytes(batch_data[offset:offset+8], byteorder='little')
            chunk_offset = int.from_bytes(batch_data[offset+8:offset+16], byteorder='little')
            chunk_size = int.from_bytes(batch_data[offset+16:offset+24], byteorder='little')
            
            self.cache[i] = (chunk_id_read, chunk_offset, chunk_size)
    
    def _load_entry(self, chunk_id: int) -> tuple:
        """Load a specific entry, using batched loading for efficiency"""
        if chunk_id in self.cache:
            return self.cache[chunk_id]
            
        if chunk_id >= self.num_chunks:
            raise IndexError(f"Chunk ID {chunk_id} out of bounds (max: {self.num_chunks-1})")
        
        # Calculate which batch this entry belongs to
        batch_start = (chunk_id // self.buffer_size) * self.buffer_size
        
        # Load the entire batch containing this entry
        self._load_entries_batch(batch_start)
        
        # The entry should now be in cache
        return self.cache[chunk_id]

class CompressedArrayStorage:
    """
    Memory-efficient storage for large sorted numeric arrays with fast random access.
    Optimized for k-mer data with 512-element chunking.
    Features incremental file saving to avoid memory issues with large arrays.
    """
    
    def __init__(
        self, 
        chunk_size: int = 512,
        compression_level: int = 9,
        compression_lib: str = 'zstd',
        dtype: np.dtype = np.uint64,
        fastpfor_codec: str = None,
        sort_chunks: bool = False,
        use_delta: bool = False
    ):
        """
        Initialize the compressed array storage.
        
        Args:
            chunk_size: Number of elements per chunk (default: 512)
            compression_level: Compression level (1-9, where 9 is highest)
            compression_lib: Compression library to use ('zstd', 'lz4', 'zlib', etc.)
            dtype: Data type of array elements
        """
        self.chunk_size = chunk_size
        self.compression_level = compression_level
        self.compression_lib = compression_lib
        self.dtype = dtype
        self.element_size = np.dtype(dtype).itemsize
        
        # Storage for compressed chunks
        self.chunks = {}
        self.chunk_offsets = {}
        self.chunk_table = None
        self.num_elements = 0
        self.is_file_backed = False
        self.filename = None
        self.file_handle = None
        self.metadata_size = 0
        self.mmap_data = None
        self.fastpfor_codec = fastpfor_codec

        self.sort_chunks = sort_chunks
        self.use_delta = use_delta
        self.sorting_enabled = {}
        
        # Initialize blosc
        blosc.set_nthreads(4)  # Adjust based on your system

    def _setup_mmap(self):
        """Set up memory mapping for the file to avoid excessive I/O operations"""
        if self.mmap_data is not None:
            return
        
        self.file_handle = open(self.filename, 'rb')
        self.mmap_data = mmap.mmap(self.file_handle.fileno(), 0, access=mmap.ACCESS_READ)
        self.in_memory_compressed = True
        print(f"File {self.filename} memory-mapped for faster access")
        
    def _get_chunk_id(self, index: int) -> int:
        """Get the chunk ID for a given element index."""
        return index // self.chunk_size
    
    def _get_offset_in_chunk(self, index: int) -> int:
        """Get the offset within a chunk for a given element index."""
        return index % self.chunk_size
        
    def load_all_chunks_to_memory(self) -> None:
        """
        Load all compressed chunks into memory for faster access.
        Chunks will still be decompressed on demand, but accessing them
        will be faster since the data is already in memory.
        """
        if not self.is_file_backed:
            raise ValueError("You must load chunk information with load() before loading chunks to memory!")
            
        if hasattr(self, 'in_memory_compressed') and self.in_memory_compressed:
            # Already loaded in memory compressed format
            return
            
        with open(self.filename, 'rb') as f:
            for chunk_id in tqdm(range(len(self.chunk_table))):
                # Skip if we already have this chunk
                if chunk_id in self.chunks:
                    continue

                chunk_offs = int(self.chunk_table[chunk_id][1])
                chunk_size = int(self.chunk_table[chunk_id][2])
                    
                # Seek to the chunk offset without the size
                f.seek(chunk_offs + 8)
                
                # Read compressed chunk
                compressed_chunk = f.read(chunk_size)
                self.chunks[chunk_id] = compressed_chunk
                
        # Update state
        self.in_memory_compressed = True
        
    def unload_chunks_from_memory(self) -> None:
        """
        Unload chunks from memory to free up space.
        The array will remain accessible but will read from disk.
        """
        if not self.is_file_backed:
            raise ValueError("Cannot unload chunks: not file-backed")
            
        # Clear chunks from memory
        self.chunks = {}
        self.in_memory_compressed = False
    
    def _compress_chunk(self, chunk: np.ndarray) -> bytes:
        """
        Compress a numpy array chunk.
        
        Args:
            chunk: Numpy array to compress
            
        Returns:
            Compressed bytes
        """
        # Ensure chunk is contiguous in memory
        if not chunk.flags.c_contiguous:
            chunk = np.ascontiguousarray(chunk.astype(self.dtype))
            
        # CRITICAL: Always ensure the chunk is exactly chunk_size elements
        # This is essential for compatibility with Cython code that expects fixed-size chunks
        if len(chunk) != self.chunk_size:
            if len(chunk) < self.chunk_size:
                # Pad with the last value (or zero for empty chunks)
                pad_size = self.chunk_size - len(chunk)
                pad_value = chunk[-1] if len(chunk) > 0 else 0
                padding = np.full(pad_size, pad_value, dtype=self.dtype)
                chunk = np.concatenate([chunk, padding])
            else:
                # Truncate if too large
                chunk = chunk[:self.chunk_size]
        
        # Verify the chunk size before compression
        assert len(chunk) == self.chunk_size, f"Chunk size mismatch: {len(chunk)} != {self.chunk_size}"
        
        original_positions = None
    
        if self.fastpfor_codec:
            chunk_u32 = chunk.astype(np.uint32) if chunk.dtype != np.uint32 else chunk
            chunk_to_compress = chunk_u32
            
            if self.sort_chunks:
                original_positions = np.arange(len(chunk_to_compress), dtype=np.uint16)
                sorted_indices = np.argsort(chunk_to_compress)
                chunk_to_compress = chunk_to_compress[sorted_indices]
                original_positions = original_positions[sorted_indices]
                self.sorting_enabled[id(chunk_to_compress)] = True
            
            applied_delta = False
            if self.use_delta and (self.sort_chunks or self.fastpfor_codec in ['simdfastpfor128', 'simdoptpfor']):
                delta_chunk = np.array(chunk_to_compress, copy=True)
                delta1(delta_chunk, len(delta_chunk))
                chunk_to_compress = delta_chunk
                applied_delta = True

            compressed = np.zeros(len(chunk_to_compress) + 1024, dtype=np.uint32)

            codec = getCodec(self.fastpfor_codec)
            comp_size = codec.encodeArray(
                chunk_to_compress, len(chunk_to_compress),
                compressed, len(compressed)
            )

            compressed = compressed[:comp_size]

            header = {
                'codec': self.fastpfor_codec,
                'delta': applied_delta,
                'sorted': self.sort_chunks,
                'comp_size': comp_size,
                'orig_size': len(chunk),
                'dtype': str(chunk.dtype)
            }
            
            header_bytes = pickle.dumps(header)
            header_size = len(header_bytes).to_bytes(8, byteorder='little')

            result = header_size + header_bytes + compressed.tobytes()

            if self.sort_chunks and original_positions is not None:
                pos_compressed = blosc.pack_array(original_positions, 
                                                  clevel=self.compression_level,
                                                  shuffle=blosc.SHUFFLE)
                result += pos_compressed
                
            return result
        else:
            original_positions = None
            chunk_to_compress = chunk
            applied_delta = False
            
            if self.sort_chunks:
                original_positions = np.arange(len(chunk), dtype=np.uint16)
                sorted_indices = np.argsort(chunk)
                chunk_to_compress = chunk[sorted_indices]
                original_positions = original_positions[sorted_indices]
            
            # Apply delta encoding if enabled and sorting was applied
            if self.use_delta and self.sort_chunks:
                delta_chunk = np.array(chunk_to_compress, copy=True)
                delta_chunk[1:] = delta_chunk[1:] - delta_chunk[:-1]
                chunk_to_compress = delta_chunk
                applied_delta = True

            if self.sort_chunks or applied_delta:
                header = {
                    'sorted': self.sort_chunks,
                    'delta': applied_delta,
                    'dtype': str(chunk.dtype)
                }
                header_bytes = pickle.dumps(header)
                header_size = len(header_bytes).to_bytes(8, byteorder='little')

                try:
                    compressed_data = blosc.pack_array(chunk_to_compress, 
                                            clevel=self.compression_level,
                                            cname=self.compression_lib,
                                            shuffle=blosc.SHUFFLE)

                    if self.sort_chunks:
                        pos_compressed = blosc.pack_array(original_positions,
                                                clevel=self.compression_level,
                                                cname=self.compression_lib,
                                                shuffle=blosc.SHUFFLE)
                        # Combine everything: header_size + header + compressed_data + pos_compressed
                        return header_size + header_bytes + compressed_data + pos_compressed
                    else:
                        return header_size + header_bytes + compressed_data
                except Exception as e:
                    # Fallback to original method
                    print(f"Warning: Advanced compression failed: {str(e)}. Falling back to basic compression.")
            
            try:
                compressed = blosc.pack_array(chunk, 
                                            clevel=self.compression_level,
                                            cname=self.compression_lib,
                                            shuffle=blosc.SHUFFLE)
                return compressed
            except Exception as e:
                try:
                    # Fall back to basic compress method which works on bytes
                    # Convert numpy array to bytes
                    chunk_bytes = chunk.tobytes()
                    
                    # Compress using basic blosc.compress
                    compressed = blosc.compress(
                        chunk_bytes,
                        typesize=chunk.dtype.itemsize,
                        clevel=self.compression_level,
                        cname=self.compression_lib,
                        shuffle=blosc.SHUFFLE
                    )
                    return compressed
                except Exception as e:
                    # If all compression methods fail, store uncompressed
                    # This is a last resort and will not be space-efficient
                    print(f"Warning: Compression failed, storing uncompressed. Error: {str(e)}")
                    # We'll use a simple prefix to indicate uncompressed data
                    return b'UNCOMP:' + chunk.tobytes()
    
    def _decompress_chunk(self, compressed_data: bytes) -> np.ndarray:
        """
        Decompress a chunk of data.
        
        Args:
            compressed_data: Compressed chunk bytes
            
        Returns:
            Numpy array with decompressed data
        """
        try:
            result = blosc.unpack_array(compressed_data)
            return result
        except Exception as e:
            raise ValueError(f"Decompression failed. Error: {str(e)}")
    
    def _start_incremental_save(self, filename: str) -> None:
        """
        Start an incremental save operation.
        
        Args:
            filename: File path to save to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Open file for writing
        self.file_handle = open(filename, 'wb')
        self.filename = filename
        
        # Reserve space for metadata position (8 bytes)
        self.file_handle.write(b'\0' * 8)
        
        # Reserve space for chunk table size (8 bytes)
        self.file_handle.write(b'\0' * 8)
        
        # Initialize chunk tracking
        self.chunk_offsets = {}
        self.chunk_sizes = {}
        self.is_file_backed = True
        
    def _finalize_incremental_save(self) -> None:
        """
        Finalize the incremental save by writing the chunk table and metadata.
        """
        if self.file_handle is None:
            raise ValueError("No file open for incremental saving")
        
        # Record end of data / beginning of chunk table
        chunk_table_pos = self.file_handle.tell()
        
        # Write chunk table (chunk_id, offset, size for each chunk)
        for chunk_id, offset in sorted(self.chunk_offsets.items()):
            self.file_handle.write(chunk_id.to_bytes(8, byteorder='little'))
            self.file_handle.write(offset.to_bytes(8, byteorder='little'))
            if chunk_id in self.chunk_sizes:
                self.file_handle.write(self.chunk_sizes[chunk_id].to_bytes(8, byteorder='little'))
            else:
                self.file_handle.write(b'\0' * 8)  # If size unknown for some reason
        
        # Record position for metadata
        metadata_pos = self.file_handle.tell()
        
        # Create and write metadata
        metadata = {
            'num_elements': self.num_elements,
            'chunk_size': self.chunk_size,
            'dtype': np.dtype(self.dtype).str,
            'compression_lib': self.compression_lib,
            'compression_level': self.compression_level,
            'num_chunks': len(self.chunk_offsets),
            'fastpfor_codec': self.fastpfor_codec,
            'sort_chunks': self.sort_chunks,
            'use_delta': self.use_delta
        }
        
        metadata_bytes = pickle.dumps(metadata)
        self.file_handle.write(metadata_bytes)
        
        # Go back to beginning and write positions
        self.file_handle.seek(0)
        self.file_handle.write(metadata_pos.to_bytes(8, byteorder='little'))
        self.file_handle.write(chunk_table_pos.to_bytes(8, byteorder='little'))
        
        # Close the file
        self.file_handle.close()
        self.file_handle = None
                
    def _save_chunk_incrementally(self, chunk_id: int, compressed_chunk: bytes) -> None:
        """
        Save a compressed chunk incrementally to the open file.
        
        Args:
            chunk_id: ID of the chunk
            compressed_chunk: Compressed chunk data
        """
        if self.file_handle is None:
            raise ValueError("No file open for incremental saving")
            
        # Store the current position in the file
        current_position = self.file_handle.tell()
        self.chunk_offsets[chunk_id] = current_position
        
        # Write chunk size and data
        chunk_size = len(compressed_chunk)
        self.chunk_sizes[chunk_id] = chunk_size
        self.file_handle.write(chunk_size.to_bytes(8, byteorder='little'))
        self.file_handle.write(compressed_chunk)
        
        # Remove from memory after saving (unless we want to keep it)
        if chunk_id in self.chunks:
            del self.chunks[chunk_id]
                
    def from_array(self, array: np.ndarray, filename: str = None) -> None:
        """
        Initialize storage from a numpy array.
        If filename is provided, save incrementally to reduce memory usage.
        
        Args:
            array: Numpy array to store
            filename: Optional file path to save to incrementally
        """ 
        self.num_elements = len(array)
        num_chunks = math.ceil(self.num_elements / self.chunk_size)
        
        # Start incremental save if filename provided
        if filename:
            self._start_incremental_save(filename)
        
        # Process each chunk with progress bar
        for chunk_id in tqdm(range(num_chunks), desc="Processing chunks"):
            start_idx = chunk_id * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, self.num_elements)
            
            # Extract the chunk
            chunk = array[start_idx:end_idx]
            
            # Compress the chunk
            compressed_chunk = self._compress_chunk(chunk)
            
            # Save incrementally if requested
            if filename:
                self._save_chunk_incrementally(chunk_id, compressed_chunk)
            else:
                # Otherwise store in memory
                self.chunks[chunk_id] = compressed_chunk
        
        # Finalize the incremental save if requested
        if filename:
            self._finalize_incremental_save()
    
    def from_generator(self, generator, total_size: int, filename: str) -> None:
        """
        Initialize storage from a generator that yields sorted chunks.
        This allows processing very large arrays without loading everything into memory.
        
        Args:
            generator: Generator that yields sorted chunks of data
            total_size: Total number of elements expected
            filename: File path to save to
        """
        self.num_elements = total_size
        num_chunks = math.ceil(self.num_elements / self.chunk_size)
        
        # Start incremental save
        self._start_incremental_save(filename)
        
        # Buffer to accumulate elements
        buffer = []
        chunk_id = 0
        
        # Process data from generator
        for chunk in tqdm(generator, desc="Processing chunks"):
            # Convert to numpy array if needed
            if not isinstance(chunk, np.ndarray):
                chunk = np.array(chunk, dtype=self.dtype)
                
            # Add to buffer
            buffer.extend(chunk)
            
            # Process complete chunks
            while len(buffer) >= self.chunk_size:
                # Extract a chunk
                chunk_data = np.array(buffer[:self.chunk_size], dtype=self.dtype)
                buffer = buffer[self.chunk_size:]
                
                # Compress and save
                compressed_chunk = self._compress_chunk(chunk_data)
                self._save_chunk_incrementally(chunk_id, compressed_chunk)
                
                chunk_id += 1
        
        # Process any remaining data
        if buffer:
            chunk_data = np.array(buffer, dtype=self.dtype)
            
            # Compress and save
            compressed_chunk = self._compress_chunk(chunk_data)
            self._save_chunk_incrementally(chunk_id, compressed_chunk)
        
        # Finalize the save
        self._finalize_incremental_save()
    
    def save(self, filename: str) -> None:
        """
        Save the compressed array to a file.
        
        Args:
            filename: File path to save to
        """
        # Create dictionary of metadata
        metadata = {
            'num_elements': self.num_elements,
            'chunk_size': self.chunk_size,
            'dtype': np.dtype(self.dtype).str,
            'compression_lib': self.compression_lib,
            'compression_level': self.compression_level
        }
        
        # Open file for writing
        with open(filename, 'wb') as f:
            # Write metadata
            pickle.dump(metadata, f)
            
            # Write chunk offsets and compressed chunks
            current_offset = f.tell()
            
            for chunk_id in range(len(self.chunks)):
                if chunk_id in self.chunks:
                    compressed_chunk = self.chunks[chunk_id]
                    
                    # Store the offset
                    self.chunk_offsets[chunk_id] = current_offset
                    
                    # Write chunk size and then chunk data
                    chunk_size = len(compressed_chunk)
                    f.write(chunk_size.to_bytes(8, byteorder='little'))
                    f.write(compressed_chunk)
                    
                    # Update offset for next chunk
                    current_offset = f.tell()
        
        # Update state
        self.is_file_backed = True
        self.filename = filename
    
    def load(self, filename: str, load_in_memory: bool = False, max_chunks_in_memory: int = 100_000_000) -> None:
        with open(filename, 'rb') as f:
            # Read metadata and chunk table positions
            metadata_pos = int.from_bytes(f.read(8), byteorder='little')
            chunk_table_pos = int.from_bytes(f.read(8), byteorder='little')
            
            # Jump to metadata first to get number of chunks
            f.seek(metadata_pos)
            metadata = pickle.load(f)
            
            # Restore attributes
            self.num_elements = metadata['num_elements']
            self.chunk_size = metadata['chunk_size']
            self.dtype = np.dtype(metadata['dtype'])
            self.compression_lib = metadata['compression_lib']
            self.compression_level = metadata['compression_level']
            self.element_size = self.dtype.itemsize
            self.fastpfor_codec = metadata.get('fastpfor_codec', None)
            self.sort_chunks = metadata.get('sort_chunks', False)
            self.use_delta = metadata.get('use_delta', False)
            num_chunks = metadata['num_chunks']
            
            if num_chunks <= max_chunks_in_memory:
                # Small enough to load in memory - use original approach
                f.seek(chunk_table_pos)
                # Each entry is 24 bytes (8 for chunk_id, 8 for offset, 8 for size)
                chunk_table_bytes = f.read(num_chunks * 24)
                
                chunk_table_dtype = np.dtype([
                    ('chunk_id', '<u8'),
                    ('offset', '<u8'),
                    ('size', '<u8')
                ])
                
                self.chunk_table = np.frombuffer(chunk_table_bytes, dtype=chunk_table_dtype)
            else:
                # Too large - use lazy loading
                print(f"Use lazy loading for {filename}")
                self.chunk_table = LazyChunkTable(filename, chunk_table_pos, num_chunks)
        
        # Update state
        self.is_file_backed = True
        self.filename = filename
        if os.path.getsize(filename) > 1e9:
            self._setup_mmap()
        elif load_in_memory:
            self.load_all_chunks_to_memory()
    
    def _load_chunk_from_file(self, chunk_id: int) -> np.ndarray:
        """
        Load a specific chunk from the backing file.
        
        Args:
            chunk_id: ID of the chunk to load
            
        Returns:
            Decompressed numpy array for the chunk
        """
        if not self.is_file_backed or self.filename is None:
            raise ValueError("No backing file available")
            
        if chunk_id >= len(self.chunk_table):
            raise IndexError(f"Chunk ID {chunk_id} not found")
        
        # Fast chunk info extraction
        chunk_entry = self.chunk_table[chunk_id]
        if isinstance(chunk_entry, tuple):
            _, chunk_offs, chunk_size = chunk_entry
        else:
            chunk_offs = int(chunk_entry[1])
            chunk_size = int(chunk_entry[2])
        
        # Direct memory access for mmapped files (no copy)
        if hasattr(self, 'mmap_data') and self.mmap_data is not None:
            return self._decompress_chunk(self.mmap_data[chunk_offs+8:chunk_offs+8+chunk_size])
            
        # Optimized file reading
        with open(self.filename, 'rb') as f:
            f.seek(chunk_offs + 8)
            return self._decompress_chunk(f.read(chunk_size))
    
    def get_chunk(self, chunk_id: int) -> np.ndarray:
        """
        Get a specific chunk by ID.
        
        Args:
            chunk_id: ID of the chunk to retrieve
            
        Returns:
            Numpy array with the chunk data
        """
        # Check if chunk is in memory (compressed format)
        if chunk_id in self.chunks:
            return self._decompress_chunk(self.chunks[chunk_id])
            
        # If file-backed, load from file
        if self.is_file_backed:
            return self._load_chunk_from_file(chunk_id)
            
        raise IndexError(f"Chunk ID {chunk_id} not found")
        
    def get_chunk_raw(self, chunk_id: int) -> bytes:
        """
        Get a specific chunk in its raw compressed format.
        This is useful for advanced use cases where you want to
        manage the decompression yourself.
        
        Args:
            chunk_id: ID of the chunk to retrieve
            
        Returns:
            Raw compressed bytes for the chunk
        """
        # Check if chunk is in memory
        if chunk_id in self.chunks:
            return self.chunks[chunk_id]
            
        # If file-backed, load raw bytes from file
        if self.is_file_backed:
            with open(self.filename, 'rb') as f:
                f.seek(self.chunk_table[chunk_id][1])
                chunk_size = int.from_bytes(f.read(8), byteorder='little')
                return f.read(chunk_size)
                
        raise IndexError(f"Chunk ID {chunk_id} not found")
    
    def get(self, index: int) -> np.number:
        """
        Get the value at a specific index.
        
        Args:
            index: Index of the element to retrieve
            
        Returns:
            Value at the specified index
        """
        if index < 0 or index >= self.num_elements:
            raise IndexError(f"Index {index} out of bounds for array of size {self.num_elements}")
            
        chunk_id = self._get_chunk_id(index)
        offset = self._get_offset_in_chunk(index)
        
        chunk = self.get_chunk(chunk_id)
        return chunk[offset]
    
    def get_batch(self, indices: List[int]) -> np.ndarray:
        """
        Get values at multiple indices efficiently.
        
        Args:
            indices: List of indices to retrieve
            
        Returns:
            Numpy array with values at the specified indices
        """
        # Group indices by chunk to minimize chunk loads
        chunk_indices = {}
        for i, idx in enumerate(indices):
            if idx < 0 or idx >= self.num_elements:
                raise IndexError(f"Index {idx} out of bounds for array of size {self.num_elements}")
                
            chunk_id = self._get_chunk_id(idx)
            offset = self._get_offset_in_chunk(idx)
            
            if chunk_id not in chunk_indices:
                chunk_indices[chunk_id] = []
                
            chunk_indices[chunk_id].append((i, offset))
        
        # Allocate result array
        result = np.zeros(len(indices), dtype=self.dtype)
        
        # Fill result array chunk by chunk
        for chunk_id, offset_pairs in chunk_indices.items():
            chunk = self.get_chunk(chunk_id)
            
            for result_idx, chunk_offset in offset_pairs:
                result[result_idx] = chunk[chunk_offset]
                
        return result

    def prefetch_chunks(self, chunk_ids):
        """
        Prefetch multiple chunks at once to reduce I/O overhead.
        Especially useful for known access patterns.
        
        Args:
            chunk_ids: List of chunk IDs to prefetch
        """
        if not self.is_file_backed:
            return
        
        for chunk_id in tqdm(chunk_ids):
            self.chunk_table[chunk_id]

    def get_multiple_slices(self, slice_requests):
        """
        Optimized implementation for multiple slice requests.
        
        Args:
            slice_requests: List of (start_idx, end_idx, kmer) tuples
        """
        result_mapping = {}
        
        all_chunks_needed = set()
        request_details = {}
        
        for start_idx, end_idx, kmer in slice_requests:
            result_size = end_idx - start_idx
            result_mapping[kmer] = np.zeros(result_size, dtype=self.dtype)

            chunk_id_start = start_idx // self.chunk_size
            chunk_id_end = (end_idx - 1) // self.chunk_size

            request_details[kmer] = (start_idx, end_idx, chunk_id_start, chunk_id_end)

            for chunk_id in range(chunk_id_start, chunk_id_end + 1):
                all_chunks_needed.add(chunk_id)

        self.prefetch_chunks(sorted(all_chunks_needed))
        
        for kmer, (start_idx, end_idx, chunk_id_start, chunk_id_end) in tqdm(request_details.items()):
            result = result_mapping[kmer]
            result_pos = 0
            
            for chunk_id in range(chunk_id_start, chunk_id_end + 1):
                chunk = self.get_chunk(chunk_id)
                
                # Calculate slice boundaries
                chunk_start = chunk_id * self.chunk_size
                local_start = max(0, start_idx - chunk_start)
                local_end = min(self.chunk_size, end_idx - chunk_start)
                
                # Copy data to result
                size = local_end - local_start
                result[result_pos:result_pos + size] = chunk[local_start:local_end]
                result_pos += size
        
        return result_mapping
    
    def get_slice(self, start_idx: int, end_idx: int) -> np.ndarray:
        """
        Get a slice of the array.
        
        Args:
            start_idx: Start index (inclusive)
            end_idx: End index (exclusive)
            
        Returns:
            Numpy array with the slice data
        """
        if start_idx < 0 or start_idx >= self.num_elements:
            raise IndexError(f"Start index {start_idx} out of bounds")
            
        if end_idx > self.num_elements:
            end_idx = self.num_elements
            
        if start_idx >= end_idx:
            return np.array([], dtype=self.dtype)
            
        # Calculate chunk range
        start_chunk = self._get_chunk_id(start_idx)
        end_chunk = self._get_chunk_id(end_idx - 1)
        
        # Handle single chunk case
        if start_chunk == end_chunk:
            chunk = self.get_chunk(start_chunk)
            start_offset = self._get_offset_in_chunk(start_idx)
            end_offset = self._get_offset_in_chunk(end_idx - 1) + 1
            return chunk[start_offset:end_offset]
            
        # Multiple chunks
        result_parts = []
        
        # First chunk
        chunk = self.get_chunk(start_chunk)
        start_offset = self._get_offset_in_chunk(start_idx)
        result_parts.append(chunk[start_offset:])
        
        # Middle chunks
        for chunk_id in range(start_chunk + 1, end_chunk):
            chunk = self.get_chunk(chunk_id)
            result_parts.append(chunk)
            
        # Last chunk
        chunk = self.get_chunk(end_chunk)
        end_offset = self._get_offset_in_chunk(end_idx - 1) + 1
        result_parts.append(chunk[:end_offset])
        
        # Combine results
        return np.concatenate(result_parts)

    @property
    def shape(self) -> tuple:
        """
        Return the shape of the array.
        
        Returns:
            Tuple representing the array dimensions (only 1D arrays supported currently)
        """
        return (self.num_elements,)
    
    def __len__(self) -> int:
        """Get the number of elements in the array."""
        return self.num_elements
    
    def __getitem__(self, index) -> Union[np.number, np.ndarray]:
        """
        Get item(s) at the specified index or slice.
        
        Args:
            index: Integer index or slice
            
        Returns:
            Value or array of values
        """
        if isinstance(index, slice):
            start = 0 if index.start is None else index.start
            stop = self.num_elements if index.stop is None else index.stop
            
            # Handle negative indices
            if start < 0:
                start = max(0, self.num_elements + start)
            if stop < 0:
                stop = max(0, self.num_elements + stop)
                
            return self.get_slice(start, stop)
        elif isinstance(index, (list, np.ndarray)):
            return self.get_batch(index)
        else:
            return self.get(index)
    
    def get_chunk_for_cython(self, chunk_id: int) -> np.ndarray:
        """
        Get a chunk in a format suitable for Cython processing.
        This ensures the returned array is:
        1. Exactly chunk_size elements
        2. C-contiguous in memory
        3. Of the correct dtype
        
        Args:
            chunk_id: ID of the chunk to retrieve
            
        Returns:
            Numpy array with the chunk data, guaranteed to be chunk_size elements
        """
        # Get the chunk
        chunk = self.get_chunk(chunk_id)
        
        # Ensure it's C-contiguous
        if not chunk.flags.c_contiguous:
            chunk = np.ascontiguousarray(chunk)
        
        # Double-check the size
        if len(chunk) != self.chunk_size:
            raise ValueError(f"Chunk size mismatch: {len(chunk)} != {self.chunk_size}")
        
        return chunk