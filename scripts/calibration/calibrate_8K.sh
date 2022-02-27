soccertrack find_calibration_parameters \
    --checkerboard_files="/Users/atom/Github/SoccerTrack/x_ignore/checkerboard_crf28.MP4" \
    --output 'parameters' \
    --fps=1 \
    --scale=10 \
    --calibration_method="fisheye" \
    # --points_to_use=100


LOG_LEVEL="DEBUG" soccertrack calibrate_from_npz \
    --input 'x_ignore/checkerboard_crf28.MP4' \
    --npzfile 'x_ignore/parameters.npz' \
    --output 'x_ignore/checkerboard_crf28_calibrated.MP4' \
    --calibration_method "fisheye" \
    --crf 28 

# Cropping the video manually isnt a bad idea.