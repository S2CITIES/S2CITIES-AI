# S2CITIES-Sdk
SDK for the integration of the ML platform to the S2CITIES Mobile Application

## APIs

This SDK contains two main APIs:
- `import_zone_config(filepath)`
- `send_signal_for_help(zone_config, video_absolute_path)`


## Overview

This SDK allows to easily send new Signal for Help Alerts to the S2CITIES 
application Backend without directly dealing with low level protocol details. 
You just have to import a configuration file data, using the `import_zone_config(filepath)` API, 
and then using it to send a new alert, leveraging the `send_signal_for_help(zone_config, video_absolute_path)` API.

### Configuration file

The configuration file is a JSON file containing both **hardcoded data**, about the Backend 
endpoints (in the 'api' key), and **custom information related to the specific camera where the software is running**,
(in the 'data' key).

Therefore, in order to make everything working, you have to specify there some details of
the cam:
- (**required**) the *zone id*, provided by the S2CITIES app system
- (optional) *address* of the cam
- (optional) a *cam* identifier
- (optional) *latitude* and *longitude* of the cam

The file `_s2cities-testzone-config--template.json` contains a template of the json file
to be used by the SDK; while the file `s2cities-testzone-config.json` contains an example
of configuration file actually used by the SDK, containing test data.

### Import zone configuration

To import the zone configuration contained in the JSON configuration file, you
just have to use the `import_zone_config(filepath)` API, providing the filepath
of the configuration file itself; you will get an object containing all the configuration
information contained in that file.

### Create a new alert

To create a new alert you first must have the configuration object mentioned in the previous
section, and use the `send_signal_for_help(zone_config, video_absolute_path)` API,
providing (in order) the object configuration object and the absolute path of the video resource 
containing the Signal for Help gesture to be uploaded to the S2CITIES app system.
The API returns the new created alert object, if everything went well.
(Note: by default, the API prints each new result on the standard output; you can prevent it by providing
the optional parameter `print_result=False`)


## Example Usage

Here an example of how to use the SDK to send a new alert, assuming to use the 
already present configuration file (`s2cities-testzone-config.json`) and assuming
the folder containing the SDK has been named `s2citiesAppSdk` (pay attention to the name you gave to it)`:

```python
# import the APIs from the SDK folder 
from s2citiesAppSdk import send_signal_for_help, import_zone_config

if __name__ == '__main__':
    # 1. import zone configuration data
    zone_config = import_zone_config("s2citiesAppSdk/s2cities-testzone-config.json")

    # ... (AI application code) ...
    # ...........................
    
    # * a new alert has to be sent *
    
    # (properly retrieve the video filepath)
    video_path = "/Users/johndoe/Desktop/hand_gesture_video.mp4"
    
    # 2. Create a new Signal For Help Alert
    new_created_alert = send_signal_for_help(zone_config, video_path)
```
