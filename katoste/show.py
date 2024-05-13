import dnaio
import numpy as np
import logging
import os
import re
import tifffile

from katoste.index import KatosteIndex
from katoste.utils import check_directory_exists

# TODO: implement caching during kmer querying? (at the index class)

class KatostePlot:
    def __init__(self, index):
        if not isinstance(index, KatosteIndex):
            raise ValueError("argument `index` has to be of type `KatosteIndex`")

        self.index = index

        # load the metadata from the index file
        self.index.open()
        self.index.close()

        self.xmax = self.index.coord_lims[1]+1
        self.ymax = self.index.coord_lims[3]+1
        self.n_spatial = self.index.n_spatial

        xy = np.unravel_index(np.arange(self.n_spatial), (self.xmax, self.ymax), order='C')
        self.xy = np.vstack(xy).T

    def scatter(self, locations, intensities):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Please install matplotlib: `pip install matplotlib`")
        plt.figure(figsize=(10, 10))
        plt.scatter(self.index_coords[0][locations],
                    -self.index_coords[1][intensities],
                    alpha=1, s=1, c=np.log(intensities))
        plt.gca().set_aspect("equal")
        plt.axis("off")


    def image(self, locations, intensities,
              render_scale: int = 1, render_smoothing: float = 1.5):
        _im_shape = np.array([self.xmax, self.ymax])
        locations = np.repeat(locations, intensities.astype(int)).astype(int)

        xy = self.xy[locations]
        im, _, _ = np.histogram2d(xy[:, 0], xy[:, 1],
                                  range=[[0, self.xmax], [0, self.ymax]],
                                  bins=tuple((_im_shape * render_scale).astype(int)))

        if render_scale != 1 or render_smoothing != 1:
            try:
                from skimage.transform import resize
                from skimage.filters import gaussian
            except ImportError:
                raise ImportError("Please install skimage: `pip install scikit-image`")

            im = gaussian(im, render_smoothing)
            im = resize(im, tuple(_im_shape))

        im = ((im / im.max()) * 255).astype(np.uint8)
        return im.T


def _run_show(args):
    kmer_index = KatosteIndex(args.index_in)
    plotter = KatostePlot(kmer_index)

    outdir_exists = check_directory_exists(args.image_out)
    if not outdir_exists:
        logging.warn("The specified output directory did not exist. Creating...")
        os.mkdir(args.image_out)

    multi_out = {}

    with dnaio.open(args.query) as f:
        logging.info(f"Opened FASTA file {args.query}")
        for i, record in enumerate(f):
            logging.info(f"[{i}/n] Querying sequence {record.name}")
            locs, ints = kmer_index.where(record.sequence, lazy_index=False)

            logging.info(f"[{i}/n] Plotting")
            im = plotter.image(locs, ints)

            if args.multichannel:
                multi_out[record.name] = im
            else:
                _out_name = re.sub(r'[^\w_. -]', '_', record.name)
                _out_file = os.path.join(args.image_out, f"katoste_{_out_name}.tif")
                tifffile.imwrite(_out_file, im, metadata={"axes": "YX", "Labels": [_out_name]}, imagej=True, bigtiff=True)
        
        if args.multichannel and len(multi_out) > 0:
            _multi_out_file = os.path.join(args.image_out, "katoste_multichannel.tif")
            logging.info(f"Saving multichannel file into {_multi_out_file}")

            multi_out_np = np.vstack([l[np.newaxis] for l in list(multi_out.values())])
            print(multi_out_np.shape)

            tifffile.imwrite(_multi_out_file, multi_out_np, metadata={"axes": "CYX", "Labels": list(multi_out.keys())}, imagej=True, bigtiff=True)
    

if __name__ == "__main__":
    from katoste.cli import get_show_parser
    args = get_show_parser().parse_args()
    _run_show()