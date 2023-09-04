import json
import sys


def _create_hand_signal_body(zone_data, new_alert_id, video_format, key):
    """
    (For internal usage only) Create the body for the signal for help request
    :param zone_data: data object as in the zone config json file
    :param new_alert_id: string representing the new alert id in the system,
     as provided by the generateVideoSignedUrl api
    :param video_format: string representing the alert video resource file extension (without the starting period)
    :param key: string representing a random key associated to the video resource
    :return: a dictionary containing the body
    """
    hand_signal_alert_body = {
        "zone_id": zone_data.get("zone_id"),
        "alert_id": new_alert_id,   # * (from previous API call) *
        "format": video_format,     # * (from previous API call) *
        "key": key                  # * (from previous API call) *
    }

    address_data = zone_data.get("address")
    if address_data is not None:
        hand_signal_alert_body["address"] = address_data

    cam_data = zone_data.get("cam")
    if cam_data is not None:
        hand_signal_alert_body["cam"] = cam_data

    latitude = zone_data.get("latitude")
    longitude = zone_data.get("longitude")

    if latitude is not None and longitude is not None:
        hand_signal_alert_body["latitude"] = latitude
        hand_signal_alert_body["longitude"] = longitude

    return hand_signal_alert_body


def _check_api_response(response):
    """
    (For internal usage) It checks the response received after an API call,
    checking its status code and printing any error
    :param response: the HTTP response as provided by the 'requests' library
    :return: a boolean representing the outcome of the request (True -> OK; False -> ERROR)
    """
    if response.status_code == 201:
        # no error
        return True

    if response.status_code == 400:
        error_desc = "Bad request"
    elif response.status_code == 500:
        error_desc = "Internal server error"
    elif response.status_code == 404:
        error_desc = "Not found"
    elif response.status_code == 405:
        error_desc = "Method not allowed"
    else:
        error_desc = ""

    # error occurred
    try:
        response_json_content = response.json()
        print("S2CITIES Request error:\n- status code: %d %s\n- content: %s" %
              (response.status_code, error_desc, json.dumps(response_json_content, indent=4)), file=sys.stderr)
    except:
        print("S2CITIES Request error:\n- status code: %d %s\n- (no other error info)" %
              (response.status_code, error_desc), file=sys.stderr)

    return False
