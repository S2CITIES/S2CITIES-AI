
from s2citiesAppSdk.utils import import_zone_config, send_signal_for_help


if __name__ == '__main__':
    print('* Started python client *\n')

    zone_config = import_zone_config("s2citiesAppSdk/s2cities-testzone-config.json")
    print("* Imported zone config * ")

    new_created_alert = send_signal_for_help(zone_config)

    print('\n* Ended python client *')
