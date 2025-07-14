import json
import datetime
import logging
import config
import threading

state_lock = threading.Lock()

_previous_state_json = None

def generate_current_state(stream_states):
    """
    Generates the JSON of the current state by aggregating data from all streams.
    """
    timestamp = datetime.datetime.now().isoformat()
    state = {
        "timestamp": timestamp,
        "streams": {}
    }
    for stream_id, persons in stream_states.items():
        state["streams"][stream_id] = {"persons": persons}
    return json.dumps(state, indent=4)

def write_current_state(state_json, lock):
    """
    Writes the current state to a JSON file in a thread-safe manner.
    """
    with lock:
        try:
            with open(config.STATE_FILE_PATH, 'w') as f:
                f.write(state_json)
        except IOError as e:
            logging.error(f"I/O error when writing current state: {e}")

def states_are_different(state1_json, state2_json):
    """
    Compares two JSON states, ignoring the timestamp.
    """
    if not state1_json or not state2_json:
        return True
    try:
        state1 = json.loads(state1_json)
        state2 = json.loads(state2_json)
        return state1.get('streams') != state2.get('streams')
    except (json.JSONDecodeError, AttributeError):
        return True

def update_state(current_stream_states):
    """
    Generates the aggregated JSON state and writes it to a file only if it has changed.
    """
    global _previous_state_json
    current_state_json = generate_current_state(current_stream_states)
    if states_are_different(current_state_json, _previous_state_json):
        write_current_state(current_state_json, state_lock)
        _previous_state_json = current_state_json
        logging.debug("State updated and written to file.")
    else:
        logging.debug("State not modified, write skipped.")