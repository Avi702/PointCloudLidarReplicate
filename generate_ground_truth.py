import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import re


SUB_CUBE_SPATIAL_SIDE_SIZE = 200  
PIXEL_RESOLUTION = 1.0             
HHDC_DIR = "/Users/avnee/PointCloudLidarReplicate/HHDC"
LAZ_DIR = "/Users/avnee/PointCloudLidarReplicate/NEON" 
OUTPUT_ARRAY_DIR = "/Users/avnee/PointCloudLidarReplicate/cbd_arrays"
OUTPUT_IMAGE_DIR = "/Users/avnee/PointCloudLidarReplicate/cbd_images"
OUTPUT_PRETREATED_DIR = "/Users/avnee/PointCloudLidarReplicate/pretreated"

print("Setting up R environment...")
lidR = importr('lidR')
terra = importr('terra')

ro.r('source("/Users/avnee/PointCloudLidarReplicate/fCBDprofile_fuelmetrics.R")')
ro.r('source("/Users/avnee/PointCloudLidarReplicate/fPCpretreatment.R")')
ro.r('current_las <- NULL')
ro.r('''
preprocess_laz <- function(laz_path, save_path=NULL) {
    library(lidR)
    
    # Check if pretreated file exists
    if (!is.null(save_path) && file.exists(save_path)) {
        cat(sprintf("DEBUG: Found pretreated file: %s. Loading directly...\\n", save_path))
        las <- tryCatch({
            readLAS(save_path)
        }, error = function(e) {
            cat(sprintf("DEBUG: Error reading pretreated LAZ: %s. Proceeding to reprocessing.\\n", e$message))
            return(NULL)
        })
        
        # If successfully loaded, return it to skip processing
        if (!is.null(las) && !is.empty(las)) {
            cat(sprintf("DEBUG: Loaded pretreated cloud with %d points.\\n", npoints(las)))
            return(las)
        }
    }
    
    cat(sprintf("DEBUG: Loading full LAZ file: %s\\n", laz_path))
    # Read the whole file
    las <- tryCatch({
        readLAS(laz_path)
    }, error = function(e) {
        cat(sprintf("DEBUG: Error reading LAZ: %s\\n", e$message))
        return(NULL)
    })
    
    if (is.empty(las)) {
        cat("DEBUG: Loaded LAS is empty.\\n")
        return(NULL)
    }
    
    cat(sprintf("DEBUG: Loaded %d points. Running pre-treatment...\\n", npoints(las)))
    
    # Pretreatment on the full cloud
    las <- tryCatch({
        # We need to treat the 'las' object as 'chunk' but fPCpretreatment might expect file path or CHUNK
        # If fPCpretreatment fails with loaded object, we might need a workaround or ensure
        # it handles in-memory LAS. Original function signature has `chunk`.
        
        # NOTE: fPCpretreatment internally calls lidR functions.
        # Ensure we don't double-classify if already classified?
        # But for NEON data, we trust re-classification might be needed.
        
        # Use simpler call if needed or wrap
        fPCpretreatment(chunk=las, classify=TRUE, LMA=100, WD=600, WD_bush=600, LMA_bush=100, H_strata_bush=2, Height_filter=80)
    }, error = function(e) { 
        cat(sprintf("DEBUG: Error in pretreatment: %s\\n", e$message))
        # Important: Return NULL if pretreatment fails so we don't process empty/bad data
        return(NULL) 
    })
    
    if (is.null(las)) {
        cat("DEBUG: Pretreatment returned NULL.\\n")
        return(NULL)
    }

    if (!is.null(las)) {
        cat(sprintf("DEBUG: Pretreatment complete. %d points remaining.\\n", npoints(las)))
        
        if (!is.null(save_path)) {
            cat(sprintf("DEBUG: Saving pretreated LAS to %s\\n", save_path))
            tryCatch({
                writeLAS(las, save_path)
            }, error = function(e) {
               cat(sprintf("DEBUG: Failed to save LAS: %s\\n", e$message)) 
            })
        }
    }
    
    return(las)
}

compute_crop_metrics <- function(las_obj, min_x, max_x, min_y, max_y, res) {
    if (is.null(las_obj)) {
        cat("DEBUG: Current LAS object is NULL.\\n")
        return(NULL)
    }
    
    # Clip rectangle from the preloaded LAS
    roi <- clip_rectangle(las_obj, min_x, min_y, max_x, max_y)
    
    if (is.empty(roi)) {
        # Return a zero-filled raster of the correct size if ROI is empty
        # We can't use pixel_metrics on empty, so we simulate the output matrix downstream
        # by returning NULL here and handling it in Python, OR return a dummy raster.
        # Returning NULL is handled by Python to create zero array.
        return(NULL)
    }
    
    # Compute Pixel Metrics
    metrics <- tryCatch({
        # 1. Compute metrics (autofits to points)
        m <- pixel_metrics(roi, ~fCBDprofile_fuelmetrics(
            X=X, Y=Y, Z=Z, Zref=Zref, gpstime=gpstime, ReturnNumber=ReturnNumber,
            Easting=Easting, Northing=Northing, Elevation=Elevation, LMA=LMA, WD=WD,
            threshold=0.02, limit_N_points=10, datatype="Pixel", omega=0.77, d=1, G=0.5
        ), res = res)
        
        # 2. Force extent to match the requested crop exactly
        # Create a template raster with the exact desired extent and resolution
        r_template <- rast(xmin=min_x, xmax=max_x, ymin=min_y, ymax=max_y, resolution=res)
        
        # Resample/Extend the computed metrics to the full template
        # 'extend' pads the missing areas (empty space around trees) with NA
        m_full <- extend(m, r_template)
        
        # Ensure it aligns perfectly (in case of slight floating point rounding, crop to exact)
        # usually extend is enough if resolution matches.
        return(m_full)
        
    }, error = function(e) { 
        cat(sprintf("DEBUG: Error in pixel_metrics: %s\\n", e$message))
        return(NULL) 
    })
    
    return(metrics)
}
       
extract_layer_as_matrix <- function(rast_obj, layer_name) {
    if (is.null(rast_obj)) return(NULL)
    
    # Convert terra SpatRaster to matrix
    # If layer not found, return NULL
    if (!layer_name %in% names(rast_obj)) {
        return(NULL)
    }
    
    # Get values and reshape
    mat <- as.matrix(rast_obj[[layer_name]], wide=TRUE)
    return(mat)
}

get_laz_bounds_r <- function(fpath) {
    library(lidR)
    library(sf)
    if (!file.exists(fpath)) return(NULL)
    
    header <- readLASheader(fpath)
    # st_bbox returns named vector: xmin, ymin, xmax, ymax
    bb <- st_bbox(header)
    return(c(bb["xmin"], bb["xmax"], bb["ymin"], bb["ymax"]))
}
''')

process_laz_crop = ro.r['extract_layer_as_matrix'] 
preprocess_laz_r = ro.r['preprocess_laz']
compute_crop_metrics_r = ro.r['compute_crop_metrics']
extract_layer_as_matrix = ro.r['extract_layer_as_matrix']
get_laz_bounds_r = ro.r['get_laz_bounds_r']


def save_image(data, path):
    """Save clean image with specific colormap and no whitespace, matching cbd_testset.py style."""
    colors = ["darkblue","darkgreen", "green","lightgreen", "yellow", "orange","red","darkred","brown"]
    cmap = mcolors.LinearSegmentedColormap.from_list("fire_risk", colors)
    plt.imsave(path, data, cmap=cmap, origin='lower', vmin=0.0, vmax=0.10)

def get_laz_bounds(laz_path):
    """Read LAZ header to get bounds using lidR (via R)"""
    bounds_vec = get_laz_bounds_r(laz_path)
    
    if bounds_vec is None or bounds_vec == ro.NULL:
         return None
    min_x = bounds_vec[0]
    max_x = bounds_vec[1]
    min_y = bounds_vec[2]
    max_y = bounds_vec[3]
    
    return {'min_x': min_x, 'max_x': max_x, 'min_y': min_y, 'max_y': max_y}

def get_tensor_dimensions(npz_path):
    """Read dimensions from the NPZ file to determine crop size."""
    try:
        with np.load(npz_path) as data:
             if len(data.files) == 0: return None
             arr = data[data.files[0]]
             shape = arr.shape[-2:]
             return float(shape[1]), float(shape[0]) 
    except Exception as e:
        print(f"Error reading NPZ {npz_path}: {e}")
        return None

def main():
    os.makedirs(OUTPUT_ARRAY_DIR, exist_ok=True)
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_PRETREATED_DIR, exist_ok=True)

    # 1. Index Source LAZ files
    print(f"Indexing LAZ files in {LAZ_DIR}...")
    laz_pattern = os.path.join(LAZ_DIR, "*.laz")
    laz_files = glob.glob(laz_pattern)
    laz_cache = {}
    
    for f in laz_files:
        try:
            print(f"  Indexing {os.path.basename(f)}...")
            bounds = get_laz_bounds(f)
            if bounds:
                laz_cache[f] = bounds
            else:
                print(f"  Skipping {os.path.basename(f)} (bad bounds)")
        except Exception as e:
            print(f"  Failed to index {f}: {e}")

    if not laz_cache:
        print("No LAZ files found or indexed. Exiting.")
        return
    hhdc_files = sorted(glob.glob(os.path.join(HHDC_DIR, "*_hhdc_casals.npz")))
    
    if not hhdc_files:
        all_npz = sorted(glob.glob(os.path.join(HHDC_DIR, "*.npz")))
        hhdc_files = [f for f in all_npz if "hhdc_casals" in os.path.basename(f)]

    print(f"\nScanning {len(hhdc_files)} HHDC tensor files (casals) to map to LAZ files...")
    
    tasks = { laz: [] for laz in laz_cache }
    
    skipped_hhdc = 0
    matched_count = 0
    
    for f in hhdc_files:
        basename = os.path.basename(f)  
        try:
            parts = basename.split('_')
            center_x = float(parts[0])
            center_y = float(parts[1])
        except:
            print(f"Skipping {basename}: Cannot parse coordinates.")
            skipped_hhdc += 1
            continue
            
        dims = get_tensor_dimensions(f)
        if not dims:
            dims = (float(SUB_CUBE_SPATIAL_SIDE_SIZE), float(SUB_CUBE_SPATIAL_SIDE_SIZE))
            
        dim_x, dim_y = dims
        target_laz = None
        for laz_path, bounds in laz_cache.items():
            if (bounds['min_x'] <= center_x <= bounds['max_x']) and \
               (bounds['min_y'] <= center_y <= bounds['max_y']):
                target_laz = laz_path
                break
        
        if target_laz:
            tasks[target_laz].append({
                'path': f,
                'basename': basename,
                'center_x': center_x,
                'center_y': center_y,
                'dim_x': dim_x,
                'dim_y': dim_y
            })
            matched_count += 1
        else:
            skipped_hhdc += 1

    print(f"Mapped {matched_count} HHDC files to LAZ sources. (Skipped {skipped_hhdc})")
    for laz_path, jobs in tasks.items():
        if not jobs:
            continue
            
        print(f"\n========== Processing {os.path.basename(laz_path)} ({len(jobs)} crops) ==========")
        
        # A. Load and Preprocess FULL LAZ
        print(f"  Step 1: Loading and Preprocessing {os.path.basename(laz_path)}...")
        laz_basename = os.path.basename(laz_path)
        base_name, ext = os.path.splitext(laz_basename)
        if "_pretreated" in base_name:
            pretreated_path = laz_path
        else:
             pretreated_filename = f"{base_name}_pretreated{ext}"
             pretreated_path = os.path.join(OUTPUT_PRETREATED_DIR, pretreated_filename)
        
        ro.r.assign("current_laz_path", laz_path)
        ro.r.assign("current_save_path", pretreated_path)
        ro.r("current_las <- preprocess_laz(current_laz_path, current_save_path)")
        is_null = ro.r("is.null(current_las)")[0]
        if is_null:
            print("  FAILED to load/preprocess LAZ. Skipping all crops for this file.")
            continue
            
        # B. Process Crops
        print("  Step 2: Processing crops...")
        file_count = 0
        
        for job in jobs:
            dim_x = job['dim_x']
            dim_y = job['dim_y']
            cx = job['center_x']
            cy = job['center_y']
            
            # Calculate Bounds
            min_x = cx - dim_x / 2.0
            max_x = cx + dim_x / 2.0
            min_y = cy - dim_y / 2.0
            max_y = cy + dim_y / 2.0
            
            try:
                metrics_raster = compute_crop_metrics_r(ro.r['current_las'], min_x, max_x, min_y, max_y, PIXEL_RESOLUTION)
                target_metric = "CBD_max"
                layer_matrix = extract_layer_as_matrix(metrics_raster, target_metric)
                target_h = int(dim_y / PIXEL_RESOLUTION)
                target_w = int(dim_x / PIXEL_RESOLUTION)
                
                if layer_matrix is None or layer_matrix == ro.NULL:
                    grid_data = np.zeros((target_h, target_w))
                else:
                    grid_data = np.array(layer_matrix)
                    grid_data[grid_data == -1] = 0
                    
                    grid_data = np.nan_to_num(grid_data, 0)
                    grid_data = np.flipud(grid_data)
                    if grid_data.shape != (target_h, target_w):
                        pass
                
                npy_path = os.path.join(OUTPUT_ARRAY_DIR, job['basename'].replace('.npz', '.npy'))
                img_path = os.path.join(OUTPUT_IMAGE_DIR, job['basename'].replace('.npz', '.png'))
                
                np.save(npy_path, grid_data)
                save_image(grid_data, img_path)
                
                file_count += 1
                if file_count % 10 == 0:
                    print(f"    Processed {file_count}/{len(jobs)}...")
                    
            except Exception as e:
                print(f"    Error processing {job['basename']}: {e}")
        ro.r("rm(current_las); gc()")
        print(f"  Finished {os.path.basename(laz_path)}.")

    print(f"\nAll tasks complete.")

if __name__ == "__main__":
    main()
