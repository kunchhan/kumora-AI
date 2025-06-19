import praw
import csv
import os
# Reddit API credentials
client_id = 'Bwu0d49RXKtY5OWkGMc4Fw'
client_secret = '4BNtM-A5X50OfGTFPjsZcysQ5emlRA'
user_agent = 'Comment_Scrap'

# Set up Reddit instance
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent)

# Choose your target post
#url = 'https://www.reddit.com/r/AskWomen/comments/aaxnlr/what_do_periods_feel_like/'
#url = 'https://www.reddit.com/r/Periods/comments/sytrl0/my_period_is_literally_ruining_my_life/'
url ='https://www.reddit.com/r/AskWomen/comments/7qhc8d/in_an_average_month_how_bad_are_your_cramps/'

submission = reddit.submission(url=url)
submission.comments.replace_more(limit=None)

# Collect comments and replies recursively
def collect_comments(comments, parent_id, post_id, results):
    for comment in comments:
        comment_id = f"c_{comment.id}"
        parent_label = f"c_{parent_id}" if parent_id else ""
        results.append((comment_id, comment.body, 'unlabeled', parent_label, post_id))
        if comment.replies:
            collect_comments(comment.replies, comment.id, post_id, results)

# Store comments
all_comments = []
collect_comments(submission.comments, parent_id=None, post_id=submission.id, results=all_comments)

# Filename to append to
filename = 'reddit_comments_threaded.csv'
file_exists = os.path.isfile(filename)

# Save/appends to CSV
with open(filename, mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(["comment_id", "comment", "label", "parent_id", "post_id"])  # header only once
    writer.writerows(all_comments)

print(f"Appended {len(all_comments)} comments from post {submission.id} to '{filename}'")