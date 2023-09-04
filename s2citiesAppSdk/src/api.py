import sys
import json

from .internals import _upload_video, _create_alert


def import_zone_config(filepath):
    """
    Import the S2CITIES zone configuration file from the specified filepath and check its correctness
    :param filepath: absolute or relative path of the json config file (extension included)
    :return: a dictionary containing all the configuration information, useful for sending an alert to the backend
    """
    # read json file
    with open(filepath, "r") as f:
        # serialize it into python dict
        config = json.load(f)

    if config is None:
        print("Error reading json config file: check the filepath", file=sys.stderr)
        exit(1)

    # checks
    api = config.get("api")
    data = config.get("data")

    if api is None or data is None:
        print("Bad Format Error: JSON config file must have as top keys 'api' and 'data'", file=sys.stderr)
        exit(1)

    base_url = api.get("base_url")
    prefix = api.get("prefix")
    endpoints = api.get("endpoints")

    if base_url is None or prefix is None or endpoints is None:
        print("Bad Format Error: JSON config file must have an 'api' object"
                " with 'base_url', 'prefix' and 'endpoints'", file=sys.stderr)
        exit(1)

    zone_id = data.get("zone_id")

    if zone_id is None:
        print("Bad Format Error: JSON config file must have a 'data' object"
               " with 'zone_id'", file=sys.stderr)
        exit(1)

    # check endpoints
    create_signal_for_help = endpoints.get("create_signal_for_help")
    generate_video_signed_url = endpoints.get("generate_video_signed_url")
    if create_signal_for_help is None or generate_video_signed_url is None:
        print("Bad Format Error: JSON config file must contain as 'endpoints': "
              "\"create_signal_for_help\", \"generate_video_signed_url\"",
              file=sys.stderr)
        exit(1)

    # check zone id
    if type(zone_id) is not str or len(zone_id) != 24 or zone_id.find(" ") != -1 \
            or zone_id.startswith("*") or zone_id.endswith("*"):
        print("Zone id error: 'zone_id' must be a valid 24 characters long hexadecimal string", file=sys.stderr)
        exit(2)

    # check address
    address = data.get("address")
    if address is not None and (type(address) is not str or address.startswith("*") or address.endswith("*")):
        print("Address error: 'address', if present, must be a valid address string;"
              " remove that field from the config file if you don't need it", file=sys.stderr)
        exit(3)

    # check cam
    cam = data.get("cam")
    if cam is not None and (type(cam) is not str or cam.startswith("*") or cam.endswith("*")):
        print("Cam error: 'cam', if present, must be a valid string;"
             " remove that field from the config file if you don't need it", file=sys.stderr)
        exit(3)

    # check latitude and longitude
    latitude = data.get("latitude")
    longitude = data.get("longitude")

    if (latitude is not None and longitude is None) or (latitude is None and longitude is not None):
        # only one of the two coordinates is provided
        print("Coordinates error: you cannot provide just one between latitude and longitude; provide both of "
              "them or none", file=sys.stderr)
        exit(3)

    if latitude is not None and (type(latitude) is not str or latitude.startswith("*") or latitude.endswith("*")):
        print("Latitude error: 'latitude', if present, must be a valid string;"
              " remove that field from the config file if you don't need it", file=sys.stderr)
        exit(3)

    if longitude is not None and (type(longitude) is not str or longitude.startswith("*") or longitude.endswith("*")):
        print("Longitude error: 'longitude', if present, must be a valid string;"
              " remove that field from the config file if you don't need it", file=sys.stderr)
        exit(3)

    # * config file format is ok *
    config["import_zone_config_ok"] = True
    print("* Imported zone config * ")
    return config


def send_signal_for_help(zone_config, video_absolute_path, print_result=True):
    """
    Send a new Signal for Help alert to the S2CITIES backend api
    :param zone_config: dictionary containing the zone configuration json file data
    :param video_absolute_path: absolute path of the video resource to be sent to the Back-end system
    :param print_result: (default: True) print the new created alert on the console or not
    :return: the new created alert, or None if an error occurred
    """
    # check if zone config file has been correctly imported
    import_zone_config_ok = zone_config["import_zone_config_ok"]

    if import_zone_config_ok is None or import_zone_config_ok is not True:
        print("* Error: alert not sent."
              " Properly import JSON zone configuration file with 'import_zone_config(filepath)'"
              " before sending a Signal for Help *",
              file=sys.stderr)
        return

    # * config file correctly imported here *

    api = zone_config["api"]
    zone_data = zone_config["data"]

    # build requests URL
    apiUrl = "%s%s" % (api["base_url"], api["prefix"])
    createSignalForHelpUrl = "%s%s" % (apiUrl, api["endpoints"]["create_signal_for_help"])
    generateVideoSignedUrl = "%s%s" % (apiUrl, api["endpoints"]["generate_video_signed_url"])

    # * 1 - VIDEO UPLOAD *
    upload_video_outcome = _upload_video(generateVideoSignedUrl, video_absolute_path, print_result)

    if upload_video_outcome is None:
        # something went wrong
        return None

    alert_id, video_format, key = upload_video_outcome

    # * 2 - CREATE ALERT *
    new_alert = _create_alert(createSignalForHelpUrl, zone_data, alert_id, video_format, key, print_result)

    if new_alert is None:
        # something went wrong
        return None

    # * both video and alert successfully managed here *
    return new_alert
