import logging
import os
import pickle
import cv2
import numpy as np
import threading
import time

import config
from face_analyzer import FaceAnalyzer
from utils import calculate_face_quality

class ProfileManager:
    def __init__(self, profiles_dir, face_analyzer):
        self.profiles_dir = profiles_dir
        self.face_analyzer = face_analyzer
        self.profiles = {}
        self.lock = threading.Lock()
        os.makedirs(self.profiles_dir, exist_ok=True)

    def load_profiles(self):
        """
        Loads all profiles from disk and applies automatic pruning.
        """
        with self.lock:
            logging.info("Starting to load/reload all profiles...")
            self.profiles = {}
            for profile_name in os.listdir(self.profiles_dir):
                profile_path = os.path.join(self.profiles_dir, profile_name)
                if os.path.isdir(profile_path):
                    try:
                        pkl_path = os.path.join(profile_path, f"{profile_name}.pkl")
                        with open(pkl_path, 'rb') as f:
                            profile_data = pickle.load(f)
                            if 'embedding' in profile_data:
                                profile_data['embedding'] = np.array(profile_data['embedding'])
                                if 'sample_embeddings' not in profile_data:
                                    profile_data['sample_embeddings'] = {}
                                self.profiles[profile_name] = profile_data
                            else:
                                logging.warning(f"Profile {profile_name} does not contain an embedding. Ignored.")
                    except FileNotFoundError:
                        logging.warning(f"PKL file not found for profile {profile_name}, ignoring.")
                    except Exception as e:
                        logging.error(f"Error loading profile {profile_name}: {e}")
            logging.info(f"Load complete. Total profiles: {len(self.profiles)}.")
            self._prune_and_refine_profiles()

    def _prune_and_refine_profiles(self):
        """
        Performs sample pruning by quality and recalculates the average profile embedding.
        """
        logging.info("Starting profile pruning and refinement...")
        for profile_name in list(self.profiles.keys()):
            profile_path = os.path.join(self.profiles_dir, profile_name)
            if not os.path.isdir(profile_path):
                continue

            try:
                profile_data = self.profiles[profile_name]
                sample_embeddings = profile_data.get('sample_embeddings', {})
                
                if not sample_embeddings:
                    logging.info(f"Profile '{profile_name}' has no sample embeddings. Attempting migration from JPG files...")
                    migrated_embeddings = {}
                    sample_files_for_migration = [f for f in os.listdir(profile_path) if f.startswith('sample_') and f.endswith('.jpg')]

                    if not sample_files_for_migration:
                        logging.warning(f"No JPG files found to migrate profile '{profile_name}'. Skipping.")
                        continue

                    for sample_filename in sample_files_for_migration:
                        file_path = os.path.join(profile_path, sample_filename)
                        try:
                            image = cv2.imread(file_path)
                            if image is None: continue
                            faces = self.face_analyzer.get_faces(image, use_detection=True)
                            if faces:
                                migrated_embeddings[sample_filename] = faces[0].embedding
                        except Exception as e:
                            logging.error(f"Error during sample migration {sample_filename}: {e}")
                    
                    if migrated_embeddings:
                        logging.info(f"Migration successful for '{profile_name}'. Added {len(migrated_embeddings)} sample embeddings.")
                        profile_data['sample_embeddings'] = migrated_embeddings
                        sample_embeddings = migrated_embeddings
                    else:
                        logging.warning(f"Migration failed for '{profile_name}'. No embeddings generated.")
                        continue

                if not sample_embeddings:
                    logging.debug(f"No sample embeddings found for '{profile_name}' after migration attempt. Unable to refine.")
                    continue

                all_sample_files = list(sample_embeddings.keys())
                
                if len(all_sample_files) <= config.MAX_SAMPLES_PER_PROFILE:
                    continue

                samples_with_quality = []
                for sample_filename, embedding in sample_embeddings.items():
                    file_path = os.path.join(profile_path, sample_filename)
                    if os.path.exists(file_path):
                        image = cv2.imread(file_path)
                        if image is None:
                            quality = 0
                        else:
                            faces = self.face_analyzer.get_faces(image, use_detection=True)
                            if not faces:
                                quality = 0
                            else:
                                quality, _ = calculate_face_quality(image, faces[0].landmark_2d_106)
                        samples_with_quality.append({'filename': sample_filename, 'quality': quality, 'embedding': embedding})
                    else:
                        logging.warning(f"Physical file not found for {sample_filename} in profile {profile_name}. It will be removed from the profile.")
                        samples_with_quality.append({'filename': sample_filename, 'quality': -1, 'embedding': embedding})

                samples_with_quality.sort(key=lambda s: s['quality'], reverse=True)
                top_samples = samples_with_quality[:config.MAX_SAMPLES_PER_PROFILE*3] if len(samples_with_quality) > config.MAX_SAMPLES_PER_PROFILE else samples_with_quality

                selected = []
                if top_samples:
                    selected.append(top_samples[0])
                    while len(selected) < config.MAX_SAMPLES_PER_PROFILE and len(selected) < len(top_samples):
                        best_idx = None
                        best_dist = -1
                        for idx, candidate in enumerate(top_samples):
                            if candidate in selected:
                                continue
                            min_dist = min(np.linalg.norm(candidate['embedding'] - s['embedding']) for s in selected)
                            if min_dist > best_dist:
                                best_dist = min_dist
                                best_idx = idx
                        if best_idx is not None:
                            selected.append(top_samples[best_idx])
                        else:
                            break
                else:
                    selected = top_samples

                keep_filenames = set(s['filename'] for s in selected)
                samples_to_delete = [s for s in samples_with_quality if s['filename'] not in keep_filenames]
                samples_to_keep = [s for s in samples_with_quality if s['filename'] in keep_filenames]

                logging.info(f"Profile '{profile_name}': {len(samples_to_delete)} low quality/variety samples will be deleted.")
                for sample in samples_to_delete:
                    try:
                        os.remove(os.path.join(profile_path, sample['filename']))
                        del sample_embeddings[sample['filename']]
                        logging.debug(f"Removed sample: {sample['filename']}")
                    except OSError as e:
                        logging.error(f"Error deleting file {sample['filename']}: {e}")

                if not samples_to_keep:
                    logging.warning(f"No samples left for profile '{profile_name}' after pruning. The profile may not work correctly.")
                    continue

                logging.info(f"Recalculating embedding for '{profile_name}' with {len(samples_to_keep)} high-quality and variety samples.")
                remaining_embeddings = np.array([s['embedding'] for s in samples_to_keep])
                new_avg_embedding = np.mean(remaining_embeddings, axis=0)
                new_avg_embedding /= np.linalg.norm(new_avg_embedding)
                profile_data['embedding'] = new_avg_embedding
                profile_data['sample_embeddings'] = {s['filename']: s['embedding'] for s in samples_to_keep}
                pkl_path = os.path.join(profile_path, f"{profile_name}.pkl")
                with open(pkl_path, 'wb') as f:
                    pickle.dump(profile_data, f)
                logging.info(f"Profile '{profile_name}' updated and saved with a new refined embedding.")

            except Exception as e:
                logging.error(f"Unexpected error during pruning and refinement of profile {profile_name}: {e}", exc_info=True)
        logging.info("Profile pruning and refinement complete.")


    def recognize_face(self, current_embedding):
        with self.lock:
            if not self.profiles:
                return None, None

            current_embedding_norm = current_embedding / np.linalg.norm(current_embedding)

            best_match = None
            highest_similarity = -1

            for name, profile_data in self.profiles.items():
                profile_embedding = profile_data['embedding']
                similarity = np.dot(current_embedding_norm, profile_embedding)
                
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = name

            if highest_similarity > config.RECOGNITION_THRESHOLD:
                return best_match, highest_similarity
            else:
                return None, None

    def promote_new_profile(self, samples):
        with self.lock:
            if not samples:
                logging.warning("Attempting to promote a profile without samples.")
                return None
            
            embeddings = np.array([s['embedding'] for s in samples])
            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding /= np.linalg.norm(avg_embedding)
            
            new_profile_id = f"user_{int(time.time())}"
            profile_path = os.path.join(self.profiles_dir, new_profile_id)
            os.makedirs(profile_path, exist_ok=True)

            profile_data = {'embedding': avg_embedding, 'id': new_profile_id, 'sample_embeddings': {}}
            
            for i, sample in enumerate(samples):
                sample_filename = f"sample_{i}.jpg"
                cv2.imwrite(os.path.join(profile_path, sample_filename), sample['face_crop'])
                profile_data['sample_embeddings'][sample_filename] = sample['embedding']

            with open(os.path.join(profile_path, f"{new_profile_id}.pkl"), 'wb') as f:
                pickle.dump(profile_data, f)

            self.profiles[new_profile_id] = profile_data
            logging.info(f"New profile '{new_profile_id}' promoted and saved.")
        self.load_profiles()

    def add_sample_to_profile(self, profile_id, embedding, face_crop):
        """
        Adds a new high-quality sample to an existing profile.
        """
        logging.info(f"Starting to add sample for profile '{profile_id}'...")
        profile_path = os.path.join(self.profiles_dir, profile_id)
        pkl_path = os.path.join(profile_path, f"{profile_id}.pkl")

        if not os.path.exists(pkl_path):
            logging.warning(f"Attempting to add sample to profile '{profile_id}', but the .pkl file does not exist.")
            return

        try:
            with self.lock, open(pkl_path, 'rb') as f:
                profile_data = pickle.load(f)

            if 'sample_embeddings' not in profile_data:
                profile_data['sample_embeddings'] = {}

            if len(profile_data['sample_embeddings']) >= config.MAX_SAMPLES_PER_PROFILE:
                logging.debug(f"Profile '{profile_id}' has reached the maximum number of samples. No sample added.")
                return

            timestamp = int(time.time() * 1000)
            sample_filename = f"sample_{timestamp}.jpg"
            sample_path = os.path.join(profile_path, sample_filename)
            cv2.imwrite(sample_path, face_crop)
            logging.info(f"New sample image saved at: {sample_path}")

            profile_data['sample_embeddings'][sample_filename] = embedding
            
            all_embeddings = np.array(list(profile_data['sample_embeddings'].values()))
            new_avg_embedding = np.mean(all_embeddings, axis=0)
            new_avg_embedding /= np.linalg.norm(new_avg_embedding)
            profile_data['embedding'] = new_avg_embedding

            with self.lock, open(pkl_path, 'wb') as f:
                pickle.dump(profile_data, f)

            logging.info(f"New sample added to profile '{profile_id}'. Total samples: {len(profile_data['sample_embeddings'])}/{config.MAX_SAMPLES_PER_PROFILE}")
            
            with self.lock:
                self.profiles[profile_id] = profile_data

        except Exception as e:
            logging.error(f"Error adding sample to profile '{profile_id}': {e}", exc_info=True)


    def _prune_and_recalculate(self, profile_id, profile_data):
        """
        Performs sample pruning for a profile and recalculates the average embedding.
        """
        logging.info(f"Starting pruning and recalculation for profile '{profile_id}'...")
        try:
            sample_embeddings = profile_data.get('sample_embeddings', {})
            
            if not sample_embeddings:
                logging.info(f"No samples to prune for profile '{profile_id}'.")
                return

            all_sample_files = list(sample_embeddings.keys())
            
            if len(all_sample_files) <= config.MAX_SAMPLES_PER_PROFILE:
                logging.info(f"Profile '{profile_id}' has a sufficient number of samples ({len(all_sample_files)}). No action required.")
                return

            samples_with_quality = []
            for sample_filename, embedding in sample_embeddings.items():
                file_path = os.path.join(self.profiles_dir, profile_id, sample_filename)
                if os.path.exists(file_path):
                    image = cv2.imread(file_path)
                    if image is None:
                        quality = 0
                    else:
                        faces = self.face_analyzer.get_faces(image, use_detection=True)
                        if not faces:
                            quality = 0
                        else:
                            quality, _ = calculate_face_quality(image, faces[0].landmark_2d_106)
                        samples_with_quality.append({'filename': sample_filename, 'quality': quality, 'embedding': embedding})
                else:
                    logging.warning(f"Physical file not found for {sample_filename} in profile {profile_id}. It will be removed from the profile.")
                    samples_with_quality.append({'filename': sample_filename, 'quality': -1, 'embedding': embedding})

            samples_with_quality.sort(key=lambda s: s['quality'], reverse=True)
            top_samples = samples_with_quality[:config.MAX_SAMPLES_PER_PROFILE*3] if len(samples_with_quality) > config.MAX_SAMPLES_PER_PROFILE else samples_with_quality

            selected = []
            if top_samples:
                selected.append(top_samples[0])
                while len(selected) < config.MAX_SAMPLES_PER_PROFILE and len(selected) < len(top_samples):
                    best_idx = None
                    best_dist = -1
                    for idx, candidate in enumerate(top_samples):
                        if candidate in selected:
                            continue
                        min_dist = min(np.linalg.norm(candidate['embedding'] - s['embedding']) for s in selected)
                        if min_dist > best_dist:
                            best_dist = min_dist
                            best_idx = idx
                    if best_idx is not None:
                        selected.append(top_samples[best_idx])
                    else:
                        break
            else:
                selected = top_samples

            keep_filenames = set(s['filename'] for s in selected)
            samples_to_delete = [s for s in samples_with_quality if s['filename'] not in keep_filenames]
            samples_to_keep = [s for s in samples_with_quality if s['filename'] in keep_filenames]

            logging.info(f"Profile '{profile_id}': {len(samples_to_delete)} low quality/variety samples will be deleted.")
            for sample in samples_to_delete:
                try:
                    os.remove(os.path.join(self.profiles_dir, profile_id, sample['filename']))
                    del sample_embeddings[sample['filename']]
                    logging.debug(f"Removed sample: {sample['filename']}")
                except OSError as e:
                    logging.error(f"Error deleting file {sample['filename']}: {e}")

            if not samples_to_keep:
                logging.warning(f"No samples left for profile '{profile_id}' after pruning. The profile may not work correctly.")
                return

            logging.info(f"Recalculating embedding for '{profile_id}' with {len(samples_to_keep)} high-quality and variety samples.")
            remaining_embeddings = np.array([s['embedding'] for s in samples_to_keep])
            new_avg_embedding = np.mean(remaining_embeddings, axis=0)
            new_avg_embedding /= np.linalg.norm(new_avg_embedding)
            profile_data['embedding'] = new_avg_embedding
            profile_data['sample_embeddings'] = {s['filename']: s['embedding'] for s in samples_to_keep}
            pkl_path = os.path.join(profile_path, f"{profile_id}.pkl")
            with open(pkl_path, 'wb') as f:
                pickle.dump(profile_data, f)
            logging.info(f"Profile '{profile_id}' updated and saved with a new refined embedding.")

        except Exception as e:
            logging.error(f"Unexpected error during pruning and refinement of profile {profile_id}: {e}", exc_info=True)