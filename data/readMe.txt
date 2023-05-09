The data contains tweet IDs and annotations in the below format:

;tweetid;stance;emotions;emotion_intensity

The emotion list labels belong to different emotions categories in order: ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust',"positive","negative"]

Similarly intensity labels are the real-value sobtained from NRC-EIL Python library for the corresponding emotions as above list

Since, we share our dataset only with the tweet ids and annotations for privacy issues, it is possible to use fetchTweetFromID.py python script to fetch the tweet objects with the tweet id.
