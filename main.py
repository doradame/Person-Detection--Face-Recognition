import cv2
from ultralytics import YOLO
import numpy as np
import time
import logging
import threading
import queue

import config
from face_analyzer import FaceAnalyzer
from profile_manager import ProfileManager
from utils import calculate_ear, calculate_face_quality
from logger_config import setup_logging
import state_manager

# Global initialization of main components
try:
    face_analyzer = FaceAnalyzer(config)
    profile_manager = ProfileManager(config.PROFILES_DIR, face_analyzer)
except Exception as e:
    logging.error(f"Fatal error during global initialization: {e}", exc_info=True)
    exit()

# Global variables and synchronization structures
GLOBAL_LOCK = threading.Lock()
save_queue = queue.Queue()
frame_queue = queue.Queue() 

# Thread for I/O management and profile maintenance
def file_io_and_pruning_worker():
    """
    Dedicated thread for I/O operations and periodic profile maintenance.
    Manages the promotion of new profiles and automatic pruning.
    """
    last_pruning_time = time.time()
    while True:
        try:
            task = save_queue.get(block=False)
            action = task.get('action')
            
            logging.debug(f"Task received from queue: {action}")

            if action == 'promote':
                profile_manager.promote_new_profile(task['data'])
            elif action == 'add_sample':
                logging.info(f"Task 'add_sample' for {task.get('profile_id')} received from queue.")
                profile_manager.add_sample_to_profile(
                    task['profile_id'], 
                    task['embedding'], 
                    task['face_crop']
                )

            save_queue.task_done()
        except queue.Empty:
            now = time.time()
            if now - last_pruning_time > config.PRUNING_INTERVAL_SECONDS:
                logging.info("--- Starting periodic profile maintenance ---")
                profile_manager.load_profiles()
                last_pruning_time = now
            time.sleep(0.1)
        except Exception as e:
            logging.error(f"Error in worker thread: {e}", exc_info=True)

def process_stream(stream_url, stream_id):
    """
    Processes a single video stream in a dedicated thread.
    Manages tracking, liveness, sample collection, and profile promotion.
    """
    logging.info(f"--- Starting processing for stream {stream_id} ({stream_url}) ---")
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        logging.error(f"[{stream_id}] Error: Unable to open stream: {stream_url}")
        return

    trackers = {}
    next_tracker_id = 0
    frame_counter = 0
    last_annotations = []

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"[{stream_id}] Frame not received, end of stream or error.")
            time.sleep(2)
            cap.open(stream_url)
            continue

        process_this_frame = config.FRAME_SKIP == 0 or (frame_counter + 1) % (config.FRAME_SKIP + 1) == 0
        if process_this_frame:
            last_annotations = []
        frame_counter += 1

        if process_this_frame:
            yolo_results = model(frame, verbose=False)
            persons_in_frame = []
            for box in yolo_results[0].boxes:
                if int(box.cls) != config.DETECT_CLASS or float(box.conf) < config.MIN_YOLO_CONFIDENCE:
                    continue
                x1_person, y1_person, x2_person, y2_person = map(int, box.xyxy[0].tolist())
                last_annotations.append({'type': 'rect', 'args': [(x1_person, y1_person), (x2_person, y2_person), (255, 128, 0), 2]})
                yolo_label = f"Person ({box.conf[0]:.2f})"
                last_annotations.append({'type': 'text', 'args': [yolo_label, (x1_person, y1_person - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2]})
                person_crop = frame[y1_person:y2_person, x1_person:x2_person]
                if person_crop.size == 0: continue
                insight_faces = face_analyzer.get_faces(person_crop)
                if not insight_faces: continue
                for face in insight_faces:
                    x1_face, y1_face, x2_face, y2_face = face.bbox.astype(int)
                    abs_x1_face, abs_y1_face = x1_person + x1_face, y1_person + y1_face
                    abs_x2_face, abs_y2_face = x1_person + x2_face, y1_person + y2_face
                    face_center = ((abs_x1_face + abs_x2_face) // 2, (abs_y1_face + abs_y2_face) // 2)
                    embedding = face.embedding
                    landmarks_abs = (face.landmark_2d_106 + np.array([x1_person, y1_person])).astype(int)
                    profile_id, similarity = profile_manager.recognize_face(embedding)
                    
                    if profile_id:
                        box_color = (0, 255, 0)
                        label = f"{profile_id} ({similarity:.2f})"
                        persons_in_frame.append({"id": profile_id, "type": "known", "name": profile_id, "confidence": float(similarity)})

                        face_crop_for_sample = frame[abs_y1_face:abs_y2_face, abs_x1_face:abs_x2_face]
                        landmarks_local = landmarks_abs - np.array([abs_x1_face, abs_y1_face])
                        quality, _ = calculate_face_quality(face_crop_for_sample, landmarks_local)
                        
                        logging.debug(f"RECOGNIZED face: {profile_id} (Similarity: {similarity:.2f}, Quality: {quality:.2f})")

                        if quality > config.QUALITY_THRESHOLD:
                            logging.info(f"Sufficient quality for {profile_id}. Queued for sample addition.")
                            save_queue.put({
                                'action': 'add_sample',
                                'profile_id': profile_id,
                                'embedding': embedding,
                                'face_crop': face_crop_for_sample.copy()
                            })
                        else:
                            logging.debug(f"Insufficient quality ({quality:.2f}) to add sample to {profile_id}. Threshold: {config.QUALITY_THRESHOLD}")

                    else:
                        matched_tracker_id = None
                        min_dist = float('inf')
                        for tracker_id, t_data in trackers.items():
                            dist = np.linalg.norm(np.array(face_center) - np.array(t_data['last_center']))
                            if dist < min_dist and dist < config.TRACKER_MAX_DISTANCE:
                                min_dist = dist
                                matched_tracker_id = tracker_id
                        if matched_tracker_id is not None:
                            trackers[matched_tracker_id]['last_center'] = face_center
                            trackers[matched_tracker_id]['unseen_frames'] = 0
                            tracker_id = matched_tracker_id
                        else:
                            tracker_id = f"{stream_id}_{next_tracker_id}"
                            next_tracker_id += 1
                            trackers[tracker_id] = {
                                'last_center': face_center, 'unseen_frames': 0,
                                'liveness_confirmed': False, 'blinks': 0,
                                'eye_is_closed': False, 'samples': [], 'status': 'Liveness Check'
                            }
                            logging.info(f"[{stream_id}] New tracker created: ID {tracker_id}")
                        t_data = trackers[tracker_id]
                        if not t_data['liveness_confirmed']:
                            box_color = (0, 165, 255)
                            left_eye_pts = landmarks_abs[35:41]
                            right_eye_pts = landmarks_abs[89:95]
                            ear = (calculate_ear(left_eye_pts) + calculate_ear(right_eye_pts)) / 2.0
                            label = f"Liveness... EAR: {ear:.2f} [{tracker_id}]"
                            is_below_threshold = ear < config.EAR_THRESHOLD
                            if is_below_threshold and not t_data['eye_is_closed']:
                                t_data['eye_is_closed'] = True
                            elif not is_below_threshold and t_data['eye_is_closed']:
                                t_data['blinks'] += 1
                                t_data['eye_is_closed'] = False
                                logging.info(f"[{stream_id}] Tracker {tracker_id}: BLINK DETECTED! ({t_data['blinks']}/{config.LIVENESS_MIN_BLINKS})")
                            label += f" Blinks: {t_data['blinks']}"
                            if t_data['blinks'] >= config.LIVENESS_MIN_BLINKS:
                                t_data['liveness_confirmed'] = True
                                t_data['status'] = 'Collecting Samples'
                                logging.info(f"[{stream_id}] Tracker {tracker_id}: Liveness confirmed.")
                        elif t_data['status'] == 'Collecting Samples':
                            box_color = (255, 255, 0)
                            label = f"Unknown [{tracker_id}]"
                            face_crop = frame[abs_y1_face:abs_y2_face, abs_x1_face:abs_x2_face]
                            landmarks_local = landmarks_abs - np.array([abs_x1_face, abs_y1_face])
                            quality, _ = calculate_face_quality(face_crop, landmarks_local)
                            if quality > config.QUALITY_THRESHOLD:
                                t_data['samples'].append({'embedding': embedding, 'face_crop': face_crop.copy()})
                                logging.info(f"[{stream_id}] Tracker {tracker_id}: High-quality sample added ({len(t_data['samples'])}/{config.PROMOTION_MIN_SAMPLES})")
                            if len(t_data['samples']) >= config.PROMOTION_MIN_SAMPLES:
                                logging.info(f"[{stream_id}] Tracker {tracker_id}: Queued for promotion.")
                                save_queue.put({'action': 'promote', 'data': t_data['samples']})
                                t_data['status'] = 'Promoted'
                        elif t_data['status'] == 'Promoted':
                            box_color = (255, 0, 255)
                            label = f"Promoted [{tracker_id}]"
                        persons_in_frame.append({"id": f"tracker_{tracker_id}", "type": "unknown", "status": t_data['status']})
                    last_annotations.append({'type': 'rect', 'args': [(abs_x1_face, abs_y1_face), (abs_x2_face, abs_y2_face), box_color, 2]})
                    last_annotations.append({'type': 'text', 'args': [label, (abs_x1_face, abs_y1_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2]})
                    for (lx, ly) in landmarks_abs:
                        last_annotations.append({'type': 'circle', 'args': [(lx, ly), 1, (0, 255, 255), -1]})
            stale_trackers = []
            for tracker_id, t_data in trackers.items():
                t_data['unseen_frames'] += 1
                if t_data['unseen_frames'] > config.TRACKER_MAX_UNSEEN or t_data['status'] == 'Promoted':
                    stale_trackers.append(tracker_id)
            for tracker_id in stale_trackers:
                logging.info(f"[{stream_id}] Removing tracker {tracker_id} (Stale or Promoted).")
                del trackers[tracker_id]
            with global_states_lock:
                global_stream_states[stream_id] = persons_in_frame
        for ann in last_annotations:
            if ann['type'] == 'rect':
                cv2.rectangle(frame, *ann['args'])
            elif ann['type'] == 'text':
                cv2.putText(frame, *ann['args'])
            elif ann['type'] == 'circle':
                cv2.circle(frame, *ann['args'])
        try:
            frame_queue.put_nowait((stream_id, frame))
        except queue.Full:
            logging.warning(f"[{stream_id}] Frame queue is full, a frame has been dropped to avoid delay.")
    cap.release()
    logging.info(f"--- Processing finished for stream {stream_id} ---")

def main():
    """
    Main function: starts stream processing threads, I/O management, and display.
    Handles status updates and controlled shutdown.
    """
    global model, stop_event, global_stream_states, global_states_lock, last_state_update_time
    setup_logging()
    logging.info(f"--- Facial Recognition Backend in use: {config.FACE_RECOGNITION_BACKEND.upper()} ---")
    logging.info("Loading YOLO model...")
    model = YOLO(config.MODEL_PATH)
    profile_manager.load_profiles()
    io_worker_thread = threading.Thread(target=file_io_and_pruning_worker, daemon=True)
    io_worker_thread.start()
    logging.info("Worker thread for I/O and maintenance started.")
    stop_event = threading.Event()
    global_stream_states = {}
    global_states_lock = threading.Lock()
    threads = []
    logging.info(f"Starting {len(config.RTSP_URLS)} stream(s)...")
    for i, rtsp_url in enumerate(config.RTSP_URLS):
        thread_name = f"StreamThread-{i}"
        thread = threading.Thread(target=process_stream, args=(rtsp_url, i), name=thread_name)
        threads.append(thread)
        thread.start()
        logging.info(f"Thread '{thread_name}' started for stream URL: {rtsp_url}")
    last_state_update_time = time.time()
    try:
        while not stop_event.is_set():
            if not all(t.is_alive() for t in threads):
                logging.warning("One or more stream threads have stopped. Exiting.")
                stop_event.set()
                break
            try:
                stream_id, frame = frame_queue.get_nowait()
                window_name = f"Stream {stream_id}"
                cv2.imshow(window_name, frame)
            except queue.Empty:
                pass
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logging.info("'q' key pressed. Starting shutdown procedure...")
                stop_event.set()
                break
            now = time.time()
            if now - last_state_update_time >= config.STATE_UPDATE_INTERVAL_SECONDS:
                with global_states_lock:
                    current_states_copy = dict(global_stream_states)
                state_manager.update_state(current_states_copy)
                last_state_update_time = now
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt detected. Starting shutdown procedure...")
        stop_event.set()
    finally:
        logging.info("Waiting for all threads to terminate...")
        for thread in threads:
            thread.join()
        logging.info("All threads terminated. Clean exit.")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()