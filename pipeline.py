"""
LiDAR Fuel Analysis Pipeline
Processes LAZ files and generates fuel metrics using R LidarForFuel package
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

print("Loading LidarForFuel functions...")
ro.r('source("/Users/avnee/LiDAR/LidarReplicate/CBD.R")')
ro.r('source("/Users/avnee/LiDAR/LidarReplicate/fPCpretreatment.R")')
# ro.r('source("/Users/avnee/LiDAR/LidarForFuel/R/ffuelmetrics.R")') # Not using this one

print("Available functions:")
ro.r('print(ls())')


fPCpretreatment = ro.r['fPCpretreatment']
fCBDprofile_fuelmetrics = ro.r['fCBDprofile_fuelmetrics']


print("Setup complete.\n")


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
    print(f"  Using official fPCpretreatment function...")
    
    try:
        # Call the actual R function from LidarForFuel with correct parameters
        las_preprocessed = fPCpretreatment(
            chunk=laz_file,
            classify=True,      # Perform ground classification if not already classified
            LMA=LMA,
            WD=WD,
            WD_bush=WD,         # Use same WD for understory
            LMA_bush=LMA,       # Use same LMA for understory
            H_strata_bush=2,    # Height threshold for bush/canopy separation
            Height_filter=80    # Maximum height to filter noise
        )
        
        return las_preprocessed
        
    except Exception as e:
        print(f"  Error in fPCpretreatment: {e}")
        print(f"  Falling back to simplified preprocessing...")
        
        # Fallback to simplified version if official function fails
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
        
        # 5. Calculate a SINGLE average sensor position for the whole file
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

def calculate_metrics(las, **kwargs):
    """Calculate fuel metrics using LidarForFuel and add spatial info to profile"""
    result = fCBDprofile_fuelmetrics(datatype=las, **kwargs)
    
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
    
    # Add X, Y, Z coordinates to profile
    # X, Y are the mean coordinates of the plot (same for all height bins)
    # Z is the height value from the H column (height bin center)
    if not profile_df.empty and las is not ro.NULL:
        try:
            # Get mean X and Y from the LAS object to represent the plot location
            # We use R to calculate the mean to avoid transferring all points to Python
            ro.r.assign("temp_las", las)
            
            # Extract all points to create a dense point cloud with metrics
            # This allows 2D/3D plotting of the points
            x_coords = np.array(las.slots['data'].rx2('X'))
            y_coords = np.array(las.slots['data'].rx2('Y'))
            z_coords = np.array(las.slots['data'].rx2('Z'))
            
            # Subsample if too many points (keep max 250k for reasonable file size)
            n_points = len(x_coords)
            if n_points > 250000:
                print(f"    Subsampling from {n_points:,} to 250,000 points...")
                indices = np.random.choice(n_points, 250000, replace=False)
                x_coords = x_coords[indices]
                y_coords = y_coords[indices]
                z_coords = z_coords[indices]
            
            # Create DataFrame with points
            points_df = pd.DataFrame({
                'X': x_coords,
                'Y': y_coords,
                'Z': z_coords
            })
            
            # Map profile metrics to points based on height
            # Determine bin width d from profile
            if len(profile_df) > 1:
                d = profile_df['H'].iloc[1] - profile_df['H'].iloc[0]
            else:
                d = 1.0
            
            # Calculate corresponding H bin for each point
            points_df['H_match'] = np.floor(points_df['Z'] / d) * d + (d/2)
            
            # Prepare profile for merging
            profile_cols = [c for c in profile_df.columns if c not in ['X', 'Y', 'Z', 'H']]
            profile_to_merge = profile_df[profile_cols + ['H']].copy()
            
            # Merge points with profile metrics
            merged_df = pd.merge(points_df, profile_to_merge, left_on='H_match', right_on='H', how='left')
            
            # Clean up
            merged_df = merged_df.drop(columns=['H_match'])
            
            # Use this detailed dataframe as the result
            profile_df = merged_df
            
        except Exception as e:
            print(f"    Warning: Could not map metrics to points: {e}")

    return profile_df



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
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_profiles = []
    all_filenames = []
    
    for i, laz_file in enumerate(laz_files, 1):
        filename = Path(laz_file).name
        print(f"[{i}/{len(laz_files)}] {filename}")
        
        try:
            print("  Preprocessing with fPCpretreatment...")
            # Use the official fPCpretreatment function from LidarForFuel
            las_processed = preprocess_with_fPCpretreatment(
                laz_file, 
                WD=kwargs.get('WD', 600.0),
                LMA=kwargs.get('LMA', 100.0)
            )

            if las_processed == ro.NULL:
                print("  Skipping file: Preprocessing failed and returned an empty point cloud.")
                continue
            
            print("  Calculating metrics...")
            profile = calculate_metrics(las_processed, **kwargs)

            # Save individual profile CSV immediately
            csv_name = filename.replace('.laz', '_profile.csv')
            csv_path = output_path / csv_name
            profile.to_csv(csv_path, index=False)
            print(f"    Saved profile to: {csv_name}")

            all_profiles.append(profile)
            all_filenames.append(filename)
            
        except Exception as e:
            print(f"  Error: {e}\n")
            continue
    
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
