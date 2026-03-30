import json
import logging
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import current_thread, main_thread

import numpy as np
from rich.progress import track


def combine_indices(combine_dir, project_uuids=None, project_id_offset=0,
                           merge_projects=False, project_id_shift=23,
                           cell_id_mask=0x007FFFFF, verbose=False):
    """
    Merge multiple indices (from different samples) into one.

    This performs a per-bucket merge: for each prefix, loads suffix buckets
    from all input indices, merges suffixes, concatenates + deduplicates
    cell lists, and writes the output bucket.

    Args:
        combine_dir: directory containing subdirectories, each with a MalvaIndex
        project_uuids: optional list of project UUIDs
        project_id_offset: offset for project IDs when embedding
        merge_projects: whether to embed project IDs into cell IDs
        project_id_shift: bit shift for project ID embedding
        cell_id_mask: mask for cell ID portion
        verbose: enable verbose logging

    Returns:
        (project_mapping, n_projects): mapping dict and project count
    """
    from malva.indexes import merge_prefix_indices

    # Find all sub-indices
    if project_uuids is not None:
        subdirs = [os.path.join(combine_dir, puuid) for puuid in project_uuids]
    else:
        subdirs = sorted([
            os.path.join(combine_dir, d)
            for d in os.listdir(combine_dir)
            if os.path.isdir(os.path.join(combine_dir, d))
            and os.path.exists(os.path.join(combine_dir, d, 'meta.json'))
        ])

    if len(subdirs) == 0:
        raise ValueError(f"No indices found in {combine_dir}")

    # Build project mapping
    project_mapping = {}
    index_dirs = []

    for idx, subdir in enumerate(subdirs):
        if not os.path.exists(os.path.join(subdir, 'meta.json')):
            logging.warning(f"Skipping {subdir}: no meta.json found")
            continue

        project_id = project_id_offset + idx
        project_uuid = os.path.basename(subdir)
        if project_uuids is not None and idx < len(project_uuids):
            project_uuid = project_uuids[idx]

        project_mapping[idx] = (project_id, project_uuid)
        index_dirs.append(subdir)

    logging.info(f"Merging {len(index_dirs)} prefix indices from {combine_dir}")

    # Perform the merge
    output_dir = os.path.join(combine_dir, '_merged_index')

    merge_prefix_indices(
        index_dirs=index_dirs,
        output_dir=output_dir,
        merge_projects=merge_projects,
        project_mapping=project_mapping if merge_projects else None,
        project_id_shift=project_id_shift,
        cell_id_mask=cell_id_mask,
        verbose=verbose,
    )

    # Move merged index to the combine_dir root
    for fname in ['pi.bin', 'suffixes.bin', 'data.bin', 'meta.json']:
        src = os.path.join(output_dir, fname)
        dst = os.path.join(combine_dir, fname)
        if os.path.exists(src):
            shutil.move(src, dst)

    # Also copy spatial coords if any sub-index has them
    for subdir in index_dirs:
        coord_path = os.path.join(subdir, 'spatial_coord.npy')
        if os.path.exists(coord_path):
            shutil.copy2(coord_path, os.path.join(combine_dir, 'spatial_coord.npy'))
            break

    # Clean up
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    logging.info(f"Combined {len(index_dirs)} indices into {combine_dir}")
    return project_mapping, len(index_dirs)


def process_group(group_idx, group_dirs, base_dir, project_uuids,
                  merge_projects, project_id_offset=0, verbose=False):
    """
    Process a group of indices for hierarchical merging.
    Creates symlinks and merges the group.
    """
    parent_dir = os.path.dirname(os.path.abspath(base_dir))
    temp_group_dir = tempfile.mkdtemp(prefix="malva_pfix_group_", dir=parent_dir)

    for folder in sorted(group_dirs):
        src = os.path.join(base_dir, folder)
        dst = os.path.join(temp_group_dir, folder)
        try:
            os.symlink(src, dst)
        except Exception:
            shutil.copytree(src, dst)

    mapping, n = combine_indices(
        temp_group_dir,
        project_uuids=None,
        project_id_offset=project_id_offset,
        merge_projects=merge_projects,
        verbose=verbose,
    )

    return group_idx, mapping, temp_group_dir


def hierarchical_combine(base_dir, project_uuids=None,
                                merge_projects=False, group_size=16,
                                threads=1, verbose=False):
    """
    Hierarchically combine many prefix indices.

    For large numbers of samples (>16), this groups them and merges
    in parallel, then does a final merge of the intermediate results.

    Args:
        base_dir: directory containing per-sample index subdirectories
        project_uuids: optional list of project UUIDs
        merge_projects: embed project IDs in cell IDs
        group_size: number of indices per group
        threads: parallel workers
        verbose: enable verbose logging
    """
    all_indices = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
        and os.path.exists(os.path.join(base_dir, d, 'meta.json'))
    ])

    if len(all_indices) == 0:
        raise ValueError(f"No indices found in {base_dir}")

    if len(all_indices) <= group_size:
        logging.info("Direct merge (within group size limit)")
        combine_indices(
            base_dir,
            project_uuids=project_uuids,
            merge_projects=merge_projects,
            verbose=verbose,
        )
        return

    logging.info(f"Hierarchical merge: {len(all_indices)} indices in groups of {group_size}")

    # Build groups
    groups = []
    current_offset = 0
    for i in range(0, len(all_indices), group_size):
        group = all_indices[i:i + group_size]
        groups.append((len(groups), group, current_offset))
        current_offset += len(group)

    # Process groups in parallel
    results = []
    with ProcessPoolExecutor(max_workers=threads) as executor:
        futures = {
            executor.submit(
                process_group, group_idx, group_dirs, base_dir,
                project_uuids, merge_projects, offset, verbose
            ): group_idx
            for group_idx, group_dirs, offset in groups
        }

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                logging.info(f"Completed group {futures[future]}")
            except Exception as e:
                logging.error(f"Error in group {futures[future]}: {e}")
                raise

    results.sort(key=lambda x: x[0])

    # Final merge of intermediate results
    intermediate_dirs = [r[2] for r in results]

    parent_dir = os.path.dirname(os.path.abspath(base_dir))
    final_temp_dir = tempfile.mkdtemp(prefix="malva_pfix_final_", dir=parent_dir)

    for idx, inter_dir in enumerate(intermediate_dirs):
        link_name = os.path.join(final_temp_dir, f"intermediate_{idx:04d}")
        try:
            os.symlink(inter_dir, link_name)
        except Exception:
            shutil.copytree(inter_dir, link_name)

    combine_indices(
        final_temp_dir,
        merge_projects=False,  # Already embedded at group level
        verbose=verbose,
    )

    # Move final result to base_dir
    for fname in ['pi.bin', 'suffixes.bin', 'data.bin', 'meta.json', 'spatial_coord.npy']:
        src = os.path.join(final_temp_dir, fname)
        dst = os.path.join(base_dir, fname)
        if os.path.exists(src):
            shutil.move(src, dst)

    # Update metadata with project mapping
    all_mappings = {}
    global_chunk_idx = 0
    for group_idx, mapping, _ in results:
        for key in sorted(mapping.keys(), key=int):
            proj_id, proj_uuid = mapping[key]
            all_mappings[str(global_chunk_idx)] = [proj_id, proj_uuid]
            global_chunk_idx += 1

    meta_path = os.path.join(base_dir, 'meta.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        meta['project_mapping'] = all_mappings
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    # Cleanup
    for temp_dir in intermediate_dirs:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    if os.path.exists(final_temp_dir):
        shutil.rmtree(final_temp_dir)

    logging.info("Hierarchical merge complete")


def _run_combine(args):
    """CLI entry point for combining prefix indices."""
    from malva.utils import check_directory_exists, check_file_exists

    if not check_directory_exists(args.index_in):
        logging.error("Base directory does not exist")
        return

    project_uuids = None
    if args.merge_projects and check_file_exists(args.uuid):
        with open(args.uuid) as f:
            project_uuids = [line.rstrip() for line in f]

    # Check if already combined
    if os.path.exists(os.path.join(args.index_in, 'meta.json')):
        logging.warning("Combined index already exists")
        return

    # Count sub-indices
    index_dirs = [
        d for d in os.listdir(args.index_in)
        if os.path.isdir(os.path.join(args.index_in, d))
        and os.path.exists(os.path.join(args.index_in, d, 'meta.json'))
    ]

    if len(index_dirs) <= 16:
        logging.info(f"Direct merge of {len(index_dirs)} indices")
        combine_indices(
            args.index_in,
            project_uuids=project_uuids,
            merge_projects=args.merge_projects,
            verbose=True,
        )
    else:
        logging.info(f"Hierarchical merge of {len(index_dirs)} indices")
        hierarchical_combine(
            args.index_in,
            project_uuids=project_uuids,
            merge_projects=args.merge_projects,
            threads=args.threads,
            verbose=True,
        )

    logging.info("SUCCESS!")
