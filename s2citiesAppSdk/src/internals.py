import requests
import sys
import json

from .utils import _check_api_response, _create_hand_signal_body


def _upload_video(generateVideoSignedUrl, video_absolute_path, print_result):
    """
    (For internal usage) Call the Backend to generate a signed URL and upload
    the specified video resource to AWS S3 service
    :param generateVideoSignedUrl: complete url of the backend endpoing to generate a video signed url
    :param video_absolute_path: absolute path of the video resource to be uploaded
    :param print_result: boolean which specifies if printing results or not
    :return: the new alert ID, the video resource format and the associated key, all to be provided
     in the create alert API call; None, otherwise
    """
    # extract video format from the file extension
    video_name_items = video_absolute_path.split(".")

    if len(video_name_items) < 2:
        print("S2CITIES Error: error extracting video format. Check the video absolute path: "
              "it must contain the file extension", file=sys.stderr)
        return None

    video_format = video_name_items[-1]

    if video_format is None or len(video_format) == 0:
        print("S2CITIES Error: error extracting video format. Check the video absolute path", file=sys.stderr)
        return None

    generateVideoRequestBody = {
        "format": video_format
    }

    # call Backend api to generate signed url for video upload
    response = requests.post(
        url=generateVideoSignedUrl,
        json=generateVideoRequestBody
    )

    # check response
    ok = _check_api_response(response)

    if not ok:
        return None

    try:
        deserialized_response = response.json()
    except:
        print("S2CITIES Error during response deserialization after video url generation request", file=sys.stderr)
        return None

    # check response correctness
    upload_signed_url = deserialized_response["upload_signed_url"]
    alert_id = deserialized_response["alert_id"]
    resource_format = deserialized_response["format"]
    key = deserialized_response["key"]

    if upload_signed_url is None or alert_id is None or resource_format is None or key is None:
        print("S2CITIES video url generation response error: \"upload_signed_url\", \"alert_id\", \"format\" and "
              "\"key\" fields must be received with this call", file=sys.stderr)
        return None

    if print_result:
        formatted_generate_signed_url_response = json.dumps(deserialized_response, indent=4)
        print("* New signed video URL requested *")
        print(formatted_generate_signed_url_response)

    # upload alert video at the generated signed url

    with open(video_absolute_path, 'rb') as finput:
        print("\n* Start uploading video... *")
        response = requests.put(upload_signed_url, data=finput)

    if response is None:
        print("S2CITIES read error: an error occurred reading the video file. "
              "Check the provided video path: it must be the video absolute path", file=sys.stderr)
        return None

    if response.status_code != 200:
        # an error occurred during video upload
        try:
            response_json_content = response.json()
            print("S2CITIES Request error during video upload:\n- status code: %d\n- content: %s" %
                  (response.status_code, json.dumps(response_json_content, indent=4)), file=sys.stderr)
        except:
            print("S2CITIES Request error during video upload:\n- status code: %d\n- (no other error info)" %
                  response.status_code, file=sys.stderr)

        return None

    # * video successfully uploaded here *
    if print_result:
        print("* Alert Video successfully uploaded *")

    # return the new Alert ID, the video resource format and the key to be sent in the next call
    return alert_id, resource_format, key

def _create_alert(createSignalForHelpUrl, zone_data, alert_id, video_format, key, print_result):
    """
    (For internal usage only) Call the Backend to generate a new Alert
    with the provided ID, passing the needed cam/zone info
    :param createSignalForHelpUrl: complete URL of the Backend endpoint to create a new alert
    :param zone_data: zone configuration object
    :param alert_id: string representing the ID of the alert that will be created
    :param video_format: string representing the alert video resource file extension (without the starting period)
    :param key: string representing a random key associated to the video resource
    :param print_result: boolean which specifies if printing results or not
    :return: the new created Alert if everything went well; None, otherwise
    """
    # create request body
    request_body = _create_hand_signal_body(zone_data, alert_id, video_format, key)

    # * execute HTTP Post request *
    response = requests.post(
        url=createSignalForHelpUrl,
        json=request_body
    )

    # check response
    ok = _check_api_response(response)

    if not ok:
        return None

    # * request successfully processed *

    try:
        created_alert = response.json()

        if print_result:
            formatted_alert = json.dumps(created_alert, indent=4)
            print("\n* New created Hand Signal alert *")
            print(formatted_alert)

        return created_alert
    except:
        print("Error during new alert json serialization", file=sys.stderr)
        return None
