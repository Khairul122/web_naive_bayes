from app.extension import db
from datetime import datetime

class TwitterScraping(db.Model):
    __tablename__ = 'twitter_scraping'

    id = db.Column(db.Integer, primary_key=True)
    tweet_id_str = db.Column(db.String(50), unique=True, nullable=False)
    conversation_id_str = db.Column(db.String(50), nullable=True)
    username = db.Column(db.String(100), nullable=True)
    user_id_str = db.Column(db.String(50), nullable=True)
    full_text = db.Column(db.Text, nullable=False)
    
    favorite_count = db.Column(db.Integer, default=0)
    retweet_count = db.Column(db.Integer, default=0)
    reply_count = db.Column(db.Integer, default=0)
    quote_count = db.Column(db.Integer, default=0)
    
    created_at = db.Column(db.DateTime, nullable=False)
    tweet_url = db.Column(db.String(255), nullable=True)
    
    is_retweet = db.Column(db.Boolean, default=False)
    is_reply = db.Column(db.Boolean, default=False)
    is_quote = db.Column(db.Boolean, default=False)
    in_reply_to_screen_name = db.Column(db.String(100), nullable=True)
    in_reply_to_tweet_id = db.Column(db.String(50), nullable=True)
    
    has_media = db.Column(db.Boolean, default=False)
    media_urls = db.Column(db.Text, nullable=True)
    media_type = db.Column(db.String(50), nullable=True)
    
    lang = db.Column(db.String(10), default='id')
    scraped_at = db.Column(db.DateTime, default=datetime.utcnow)
    scraped_by = db.Column(db.Integer, db.ForeignKey('users.id_user'), nullable=False)

    def __repr__(self):
        return f'<TwitterScraping {self.tweet_id_str}>'