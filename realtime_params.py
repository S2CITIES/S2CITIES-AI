window = 2.5  # in seconds
interval = 1  # in seconds
num_predictions_ignore = 10  # number of predictions to ignore after an alert
frame_rate = 12  # in frames per second, the rate at which the video is processed
video_alert_duration = 8  # in seconds, the duration of the video alert
video_alert_post_duration = 2  # in seconds, the extra time to record after the positive (included in video_alert_duration)
results_queue_length = 3  # number of results to keep in the queue
results_queue_tolerance = 2  # number of positives to have in the queue to alert
