from app.extension import db
from datetime import datetime

class SentimentAnalysis(db.Model):
    __tablename__ = 'sentiment_analysis'

    id = db.Column(db.Integer, primary_key=True)
    tweet_id = db.Column(db.Integer, db.ForeignKey('twitter_scraping.id'), nullable=False)
    preprocessing_id = db.Column(db.Integer, db.ForeignKey('text_preprocessing.id'), nullable=True)
    
    username = db.Column(db.String(100), nullable=True)
    tweet_text = db.Column(db.Text, nullable=False)
    processed_text = db.Column(db.Text, nullable=True)
    
    sentiment_label = db.Column(db.Enum('positif', 'negatif', 'netral'), nullable=True)
    confidence_score = db.Column(db.Float, default=0.0)
    
    positive_keywords = db.Column(db.Text, nullable=True)
    negative_keywords = db.Column(db.Text, nullable=True)
    neutral_keywords = db.Column(db.Text, nullable=True)
    
    labeling_method = db.Column(db.Enum('auto', 'manual'), default='auto')
    labeled_at = db.Column(db.DateTime, nullable=True)
    labeled_by = db.Column(db.Integer, db.ForeignKey('users.id_user'), nullable=False)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<SentimentAnalysis {self.id}: {self.sentiment_label}>'


class SentimentSettings(db.Model):
    __tablename__ = 'sentiment_settings'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id_user'), nullable=False, unique=True)

    threshold_pos = db.Column(db.Float, default=0.5)
    threshold_neg = db.Column(db.Float, default=-0.5)
    labeling_engine = db.Column(db.Enum('inset', 'keyword'), default='inset')

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<SentimentSettings user={self.user_id} pos={self.threshold_pos}>'