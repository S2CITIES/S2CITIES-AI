import sys
import json
import requests

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
        if create_signal_for_help is None:
            print("Bad Format Error: JSON config file must contain as 'endpoints': \"create_signal_for_help\"",
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

        # * config file format is ok *
        config["import_zone_config_ok"] = True

        return config


def send_signal_for_help(zone_config, print_result=True):
    """
    Send a new Signal for Help alert to the S2CITIES backend api
    :param zone_config: dictionary containing the zone configuration json file data
    :param print_result: (default: True) print the new created alert on the console or not
    :return: the new created alert, or None if an error occurred
    """
    # check zone config file has been correctly imported
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

    # build request URL
    apiUrl = "%s%s" % (api["base_url"], api["prefix"])
    createSignalForHelpUrl = "%s%s" % (apiUrl, api["endpoints"]["create_signal_for_help"])

    # create request body
    request_body = create_hand_signal_body(zone_data)

    # * execute HTTP Post request *
    response = requests.post(
        url=createSignalForHelpUrl,
        json=request_body
    )

    # check status code
    if response.status_code != 201:
        if response.status_code == 400:
            error_desc = "Bad request"
        elif response.status_code == 500:
            error_desc = "Internal server error"
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

        return None

    # * request successfully processed *

    try:
        created_alert = response.json()

        if print_result:
            formatted_alert = json.dumps(created_alert, indent=4)
            print("* New created Hand Signal alert *")
            print(formatted_alert)

        return created_alert
    except:
        print("Error during new alert json serialization", file=sys.stderr)
        return None


def create_hand_signal_body(zone_data):
    """
    (For private usage) Create the body for the signal for help request
    :param zone_data: data object as in the zone config json file
    :return: a dictionary containing the body
    """
    hand_signal_alert_body = {
        "zone_id": zone_data.get("zone_id")
    }

    address_data = zone_data.get("address")
    if address_data is not None:
        hand_signal_alert_body["address"] = address_data

    cam_data = zone_data.get("cam")
    if cam_data is not None:
        hand_signal_alert_body["cam"] = cam_data

    return hand_signal_alert_body
