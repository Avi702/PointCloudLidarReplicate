"""
LiDAR Fuel Analysis Pipeline
Processes LAZ files and generates fuel metrics using R LidarForFuel package
Usage: python or python3 pipeline.py <file1.laz> [file2.laz ...]
"""


import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import pandas as pd
import numpy as np
from pathlib import Path
import sys

print("Loading R packages...")
lidR = importr('lidR')
base = importr('base')
ggplot2 = importr('ggplot2')
grDevices = importr('grDevices')
data_table = importr('data.table')
ggthemes = importr('ggthemes')
gridExtra = importr('gridExtra')
lubridate = importr('lubridate')
RANN = importr('RANN')
sf = importr('sf')
terra = importr('terra')
stringr = importr('stringr')


ro.r('source("/Users/avnee/PointCloudLidarReplicate/fCBDprofile_fuelmetrics.R")')
ro.r('source("/Users/avnee/PointCloudLidarReplicate/ffuelmetrics.R")')
ro.r('source("/Users/avnee/PointCloudLidarReplicate/fPCpretreatment.R")')




fPCpretreatment = ro.r['fPCpretreatment']
fCBDprofile_fuelmetrics = ro.r['fCBDprofile_fuelmetrics']


print("Setup complete.\n")



OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True, parents=True)
PRETREATED_DIR = OUTPUT_DIR / "pretreated"
PRETREATED_DIR.mkdir(exist_ok=True, parents=True)


ro.r('current_las_pretreated <- NULL')


def preprocess_with_fPCpretreatment(laz_file, WD=600.0, LMA=100.0):
    """
    Preprocess LAZ file using the official fPCpretreatment function from LidarForFuel.
    This is the proper way to preprocess point clouds for fuel metrics analysis.
    
    Parameters:
    - laz_file: Path to the LAZ file
    - WD: Wood density (kg/m³), default 600
    - LMA: Leaf Mass Area (g/m²), default 100
    
    Returns:
    - Preprocessed LAS object ready for fCBDprofile_fuelmetrics
    """
    laz_name = Path(laz_file).stem
    
    if laz_name.endswith('_pretreated'):
        pretreated_path = PRETREATED_DIR / f"{laz_name}.laz"
    else:
        pretreated_path = PRETREATED_DIR / f"{laz_name}_pretreated.laz"
    
    if pretreated_path.exists():
        print(f"  Found pretreated file at {pretreated_path}. Loading directly...")
        try:
            las_preprocessed = lidR.readLAS(str(pretreated_path))
             # Check if loaded LAS is valid and not empty
            ro.r.assign("temp_las", las_preprocessed)
            is_empty = ro.r("is.empty(temp_las)")[0]
            if not is_empty:
                print(f"  Successfully loaded pretreated file.")
                return las_preprocessed
            else:
                print("  Pretreated file was empty. Reprocessing...")
        except Exception as e:
             print(f"  Error loading pretreated file: {e}. Reprocessing...")

    
    print(f"  Using official fPCpretreatment function...")
    
    try:
        las_preprocessed = fPCpretreatment(
            chunk=laz_file,
            classify=True,     
            LMA=LMA,
            WD=WD,
            WD_bush=WD,         
            LMA_bush=LMA,      
            H_strata_bush=2,   
            Height_filter=80   
        )
        if las_preprocessed is not None and not ro.r("is.null(las_preprocessed)")[0]:
            print(f"  Saving pretreated file to {pretreated_path}...")
            try:
                lidR.writeLAS(las_preprocessed, str(pretreated_path))
            except Exception as e:
                print(f"  Warning: Could not save pretreated file: {e}")

        return las_preprocessed
        
    except Exception as e:
        print(f"  Error in fPCpretreatment: {e}")
        print(f"  Falling back to simplified preprocessing...")
        las = lidR.readLAS(laz_file)
        return preprocess_inmemory_fallback(las, WD, LMA)


def preprocess_inmemory_fallback(las, WD=600.0, LMA=100.0):
    """
    Simplified fallback preprocessing function.
    Only used if the official fPCpretreatment function fails.
    """
    ro.r('''
    preprocess_inmem <- function(las, wd, lma) {
        library(lidR)
        
        # 1. Save original Z coordinates
        las@data$Zref <- las@data$Z
        
        # 2. Ground classification and height normalization
        las <- classify_ground(las, csf())
        las <- normalize_height(las, tin())
        
        # 3. Filter points where normalization failed
        las <- filter_poi(las, !is.na(Z))
        
        # 4. Check for empty point cloud
        if (nrow(las@data) == 0) {
            warning("Preprocessing resulted in an empty point cloud. Returning NULL.")
            return(NULL)
        }
        
        n_points <- nrow(las@data)
        
      
        flight_altitude <- max(las@data$Z, na.rm = TRUE) + 800 
        las@data$Easting <- rep(mean(las@data$X, na.rm = TRUE), n_points)
        las@data$Northing <- rep(mean(las@data$Y, na.rm = TRUE), n_points)
        las@data$Elevation <- rep(flight_altitude, n_points)
        
        # Add other required attributes
        las@data$LMA <- rep(lma, n_points)
        las@data$WD <- rep(wd, n_points)
        
        if (!"gpstime" %in% names(las@data)) {
            las@data$gpstime <- seq_len(n_points)
        }
        
        return(las)
    }
    ''')
    return ro.r['preprocess_inmem'](las, WD, LMA)

def calculate_metrics(pixel_metric, **kwargs):
    """Calculate fuel metrics using LidarForFuel and add spatial info to profile"""
    result = fCBDprofile_fuelmetrics(datatype=pixel_metric, **kwargs)
    
    metrics_r = ro.r('`[[`')(result, 1)
    profile_r = ro.r('`[[`')(result, 2)
    
    metrics = {}
    metrics_array = np.array(metrics_r)
    names_vec = ro.r('names')(metrics_r)
    
    if names_vec is not ro.NULL:
        for i, name in enumerate(list(names_vec)):
            try:
                value = metrics_array[i]
                metrics[name] = float(value) if isinstance(value, (np.integer, np.floating)) else value
            except:
                metrics[name] = None
    

    profile_df = pd.DataFrame()
    if profile_r is not ro.NULL:
        try:
            with localconverter(ro.default_converter + pandas2ri.converter):
                profile_df = pandas2ri.rpy2py(profile_r)
        except:
            try:
                profile_df = pd.DataFrame({
                    'H': list(ro.r('`$`')(profile_r, 'H')),
                    'PAD': list(ro.r('`$`')(profile_r, 'PAD')),
                    'CBD': list(ro.r('`$`')(profile_r, 'CBD')),
                    'NRD': list(ro.r('`$`')(profile_r, 'NRD'))
                })
            except:
                pass
    
  
    if not profile_df.empty and las is not ro.NULL:
        try:
            ro.r.assign("temp_las", las)
            x_coords = np.array(las.slots['data'].rx2('X'))
            y_coords = np.array(las.slots['data'].rx2('Y'))
            z_coords = np.array(las.slots['data'].rx2('Z'))
            
            
            points_df = pd.DataFrame({
                'X': x_coords,
                'Y': y_coords,
                'Z': z_coords
            })
            

            if len(profile_df) > 1:
                d = profile_df['H'].iloc[1] - profile_df['H'].iloc[0]
            else:
                d = 1.0
            
 
            points_df['H_match'] = np.floor(points_df['Z'] / d) * d + (d/2)

            profile_cols = [c for c in profile_df.columns if c not in ['X', 'Y', 'Z', 'H']]
            profile_to_merge = profile_df[profile_cols + ['H']].copy()
            
            # Merge points with profile metrics
            merged_df = pd.merge(points_df, profile_to_merge, left_on='H_match', right_on='H', how='left')
            
            merged_df = merged_df.drop(columns=['H_match'])
            
            profile_df = merged_df
            
        except Exception as e:
            print(f"    Warning: Could not map metrics to points: {e}")

    return profile_df, metrics



def generate_pixel_metrics_grid(las, res=10.0, output_path=None, **kwargs):
    """
    Generate raster metrics for the entire point cloud using lidR::pixel_metrics
    """
    print(f"  Generating raster metrics grid (Resolution: {res}m)...")
    ro.r('''
    compute_grid <- function(las, res, wd, lma) {
        library(lidR)
        library(terra)
        
        # Ensure WD and LMA are present in LAS data if not already (safeguard)
        # This handles cases where fPCpretreatment might not have added them, or scoping issues
        if (is.null(las@data$WD)) { 
            las@data$WD <- rep(wd, npoints(las)) 
        }
        if (is.null(las@data$LMA)) { 
            las@data$LMA <- rep(lma, npoints(las)) 
        }

        # Call pixel_metrics with the formula
        # We use WD=WD and LMA=LMA to refer to the columns in the LAS object
        
        metrics <- lidR::pixel_metrics(
            las,
            func = ~fCBDprofile_fuelmetrics(
                datatype="Pixel",
                X=X, Y=Y, Z=Z, Zref=Zref,
                ReturnNumber=ReturnNumber,
                Easting=Easting, Northing=Northing, Elevation=Elevation,
                LMA=LMA, gpstime=gpstime,
                WD=WD,
                threshold=0.02,
                limit_N_points=100,
                omega=0.77,
                d=1,
                G=0.5
            ),
            res = res
        )
        return(metrics)
    }
    ''')
    
    try:
        grid = ro.r['compute_grid'](las, res, kwargs.get('WD', 600.0), kwargs.get('LMA', 100.0))
        
        if output_path and grid is not ro.NULL:
             # Save to TIF
             ro.r.assign("temp_grid", grid)
             ro.r.assign("temp_path", str(output_path))
             ro.r('terra::writeRaster(temp_grid, temp_path, overwrite=TRUE)')
             print(f"    Saved TIF to: {output_path}")
             return True
        return False
        
    except Exception as e:
        print(f"    Error generating grid: {e}")
        return False

def run_pipeline(laz_files, output_dir='results', pixel_resolution=10.0, **kwargs):
    """Process LAZ files and generate metrics and plots"""
    
    print("=" * 70)
    print("LIDAR FUEL ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"Processing {len(laz_files)} files:")
    for f in laz_files:
        print(f"  - {Path(f).name}")
    print()
    
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_dir
        
    print(f"Output directory: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_profiles = []
    all_metrics = []
    all_filenames = []
    
    for i, laz_file in enumerate(laz_files, 1):
        filename = Path(laz_file).name
        print(f"[{i}/{len(laz_files)}] {filename}")
        
        try:
            print("  Preprocessing with fPCpretreatment...")
            las_processed = preprocess_with_fPCpretreatment(
                laz_file, 
                WD=kwargs.get('WD', 600.0),
                LMA=kwargs.get('LMA', 100.0)
            )

            if las_processed == ro.NULL:
                print("  Skipping file: Preprocessing failed and returned an empty point cloud.")
                continue
            tif_name = filename.replace('.laz', '_fuel_metrics.tif')
            tif_path = output_path / tif_name
            
            success = generate_pixel_metrics_grid(
                las_processed, 
                res=pixel_resolution, 
                output_path=tif_path,
                **kwargs
            )
            
            if success:
                print(f"  Successfully processed {filename}")
            else:
                print(f"  Failed to generate metrics for {filename}")
            
        except Exception as e:
            print(f"  Error: {e}\n")
            continue

    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = output_path / 'all_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
        print(f"  Saved aggregated metrics to: {metrics_path.name}")

    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Results: {output_path}")
    print(f"Files processed: {len(all_profiles)}")
    print("=" * 70)
    return all_profiles


if __name__ == '__main__':
    from glob import glob
    
    if len(sys.argv) > 1:
        laz_files = sys.argv[1:]
    else:
        laz_files = glob('/Users/avnee/LiDAR/LidarReplicate/data/*.laz')

    if not laz_files:
        print("No LAZ files found. Usage: python3 pipeline.py <file1.laz> [file2.laz ...]")
    else:
        profiles = run_pipeline(
            laz_files,
            output_dir='results',
            WD=600.0,
            LMA=100.0,
            threshold=0.02,
            omega=0.77,
            d=1.0,
            G=0.5,
            limit_flightheight=0,
            limit_N_points=0
        )
        
        print("\nAll done. Check results/ for outputs.")
