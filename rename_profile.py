import os
import sys
import logging
import config

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def rename_profile(old_name, new_name):
    """
    Safely renames a profile by changing the name of both the folder
    and the associated .pkl file.
    """
    profiles_dir = config.PROFILES_DIR
    old_profile_path = os.path.join(profiles_dir, old_name)
    new_profile_path = os.path.join(profiles_dir, new_name)
    old_pkl_path = os.path.join(old_profile_path, f"{old_name}.pkl")
    new_pkl_path = os.path.join(old_profile_path, f"{new_name}.pkl")

    if not os.path.isdir(old_profile_path):
        logging.error(f"The profile folder '{old_name}' does not exist in '{profiles_dir}'.")
        return

    if not os.path.isfile(old_pkl_path):
        logging.error(f"The file '{old_name}.pkl' was not found inside the profile folder.")
        return

    if os.path.exists(new_profile_path):
        logging.error(f"A profile with the name '{new_name}' already exists. Choose a different name.")
        return

    try:
        logging.info(f"Renaming the pkl file from '{os.path.basename(old_pkl_path)}' to '{os.path.basename(new_pkl_path)}'...")
        os.rename(old_pkl_path, new_pkl_path)

        logging.info(f"Renaming the folder from '{old_name}' to '{new_name}'...")
        os.rename(old_profile_path, new_profile_path)
        
        logging.info(f"Profile '{old_name}' is now '{new_name}'.")

    except OSError as e:
        logging.error(f"Error during renaming: {e}")
        # Attempt to rollback in case of partial error
        if not os.path.exists(old_pkl_path) and os.path.exists(new_pkl_path):
            os.rename(new_pkl_path, old_pkl_path)
            logging.warning("Attempting to restore the original .pkl file name.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 rename_profile.py <old_name> <new_name>")
        print("Example: python3 rename_profile.py user_1752427972 Mario_Rossi")
    else:
        rename_profile(sys.argv[1], sys.argv[2])
