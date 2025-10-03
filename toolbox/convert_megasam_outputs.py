import os
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Extract MegaSAM npz data and save to npy files.")
    parser.add_argument("--megasam_npz_path", type=str, required=True,
                        help="Path to the MegaSAM .npz file")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output directory to save .npy files")
    args = parser.parse_args()

    # Load npz file
    npz_file = os.path.join(args.megasam_npz_path)
    data = np.load(npz_file)

    # Ensure output directory exists
    cam_info_dir = os.path.join(args.output_path, "cam_info")
    os.makedirs(cam_info_dir, exist_ok=True)

    # Save arrays
    np.save(os.path.join(cam_info_dir, "megasam_depth.npy"), data["depths"])
    np.save(os.path.join(cam_info_dir, "megasam_intrinsics.npy"), data["intrinsic"])
    np.save(os.path.join(cam_info_dir, "megasam_c2ws.npy"), data["cam_c2w"])

    print(f"Saved depth, intrinsics, and cam_c2w to {cam_info_dir}")

if __name__ == "__main__":
    main()
