from app.extension import db
from datetime import datetime

class TfidfConversion(db.Model):
    __tablename__ = 'tfidf_conversion'

    id = db.Column(db.Integer, primary_key=True)
    sentiment_id = db.Column(db.Integer, db.ForeignKey('sentiment_analysis.id'), nullable=False)
    
    text_input = db.Column(db.Text, nullable=False)
    feature_vector = db.Column(db.Text, nullable=False)
    feature_names = db.Column(db.Text, nullable=True)
    
    total_features = db.Column(db.Integer, default=0)
    max_features = db.Column(db.Integer, default=1000)
    min_df = db.Column(db.Float, default=0.01)
    max_df = db.Column(db.Float, default=0.95)
    
    conversion_method = db.Column(db.String(50), default='tfidf')
    converted_at = db.Column(db.DateTime, default=datetime.utcnow)
    converted_by = db.Column(db.Integer, db.ForeignKey('users.id_user'), nullable=False)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<TfidfConversion {self.id}>'

class TfidfVocabulary(db.Model):
    __tablename__ = 'tfidf_vocabulary'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id_user'), nullable=False)
    
    term = db.Column(db.String(100), nullable=False)
    feature_index = db.Column(db.Integer, nullable=False)
    document_frequency = db.Column(db.Integer, default=0)
    idf_score = db.Column(db.Float, default=0.0)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<TfidfVocabulary {self.term}>'