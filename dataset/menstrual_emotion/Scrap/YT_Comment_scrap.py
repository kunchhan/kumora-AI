import requests
import csv
import os

CSV_FILENAME = 'youtube_comments_full.csv'
API_KEY = 'AIzaSyD8OMY9QXfRAwvYpc-ukEUD4IOUSzcqf10'  # Replace with your actual YouTube Data API v3 key
#VIDEO_ID = 'BFa2egx-jI8'  # Replace with the target video ID
VIDEO_ID = 'vp-pJXhWwhg'  # Replace with the target video ID


def get_comments_and_replies(video_id, api_key):
    all_data = []
    base_url = 'https://www.googleapis.com/youtube/v3/commentThreads'
    replies_url = 'https://www.googleapis.com/youtube/v3/comments'

    params = {
        'part': 'snippet',
        'videoId': video_id,
        'key': api_key,
        'textFormat': 'plainText',
        'maxResults': 100
    }

    comment_counter = 1

    while True:
        response = requests.get(base_url, params=params).json()

        for item in response.get('items', []):
            top_comment_snippet = item['snippet']['topLevelComment']['snippet']
            top_comment = top_comment_snippet['textDisplay']
            comment_id = f"{video_id}_C{comment_counter}"
            comment_counter += 1

            # Save top-level comment
            all_data.append((comment_id, top_comment, 'unlabeled', "", video_id))

            total_replies = item['snippet'].get('totalReplyCount', 0)
            if total_replies > 0:
                parent_id = item['snippet']['topLevelComment']['id']
                reply_params = {
                    'part': 'snippet',
                    'parentId': parent_id,
                    'key': api_key,
                    'textFormat': 'plainText',
                    'maxResults': 100
                }

                while True:
                    reply_response = requests.get(replies_url, params=reply_params).json()

                    for reply in reply_response.get('items', []):
                        reply_text = reply['snippet']['textDisplay']
                        all_data.append((comment_id, reply_text, 'unlabeled', comment_id, video_id))

                    if 'nextPageToken' in reply_response:
                        reply_params['pageToken'] = reply_response['nextPageToken']
                    else:
                        break

        if 'nextPageToken' in response:
            params['pageToken'] = response['nextPageToken']
        else:
            break

    return all_data

def save_to_csv(data, filename=CSV_FILENAME):
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write header only if file doesn't exist
        if not file_exists:
            writer.writerow(["comment_id", "comment", "label", "parent_id", "video_id"])

        writer.writerows(data)

    print(f"Appended {len(data)} comments and replies to '{filename}'.")

# Run
if __name__ == "__main__":
    data = get_comments_and_replies(VIDEO_ID, API_KEY)
    save_to_csv(data)
