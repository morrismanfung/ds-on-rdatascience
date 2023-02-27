# Author: Morris M. F. Chan
# 2023-02-26

'''
This script scrapes all the hot posts from the subreddit r/datascience using the library praw.

Usage: 01-scrapig.py --date=<date>

Options:
--date=<date>   A time stamp for the data file
'''

import praw
import pandas as pd
import json

from docopt import docopt

opt = docopt(__doc__)
def main(date):
    with open('credential.json') as f:
        credential = json.load(f)

    reddit = praw.Reddit(client_id=credential['client_id'],
                        client_secret=credential['client_secret'],
                        user_agent=credential['user_agent'],
                        username=credential['username'],
                        password=credential['password'])

    hot_posts = reddit.subreddit('datascience').hot(limit=None)

    title = []
    score = []
    id = []
    url = []
    num_comments = []
    created = []
    selftext = []
    comments = []

    comment_template = {
        'id': '',
        'parent_id': '',
        'body': '',
        'link_id': ''
    }

    for post in hot_posts:
        title.append(post.title)
        score.append(post.score)
        id.append(post.id)
        url.append(post.url)
        num_comments.append(post.num_comments)
        created.append(post.created)
        selftext.append(post.selftext)

        post.comments.replace_more(limit=0) # To deactivate the "Show More" button which will cause an error
        comments_in_post = []
        for comment in post.comments.list():
            comment_dict = comment_template.copy()
            comment_dict['id'] = comment.id
            comment_dict['parent_id'] = comment.parent_id
            comment_dict['body'] = comment.body
            comment_dict['link_id'] = comment.link_id
            comments_in_post.append(comment_dict)
        comments.append(comments_in_post)

    data = pd.DataFrame({
        'title': title,
        'score': score,
        'id': id,
        'url': url,
        'num_comments': num_comments,
        'created': created,
        'selftext': selftext,
        'comments': comments
    })

    data.to_json(f'data/raw_{date}.json')

if __name__ == '__main__':
    main(opt['--date'])