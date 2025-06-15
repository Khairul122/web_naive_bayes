from app.extension import db
from datetime import datetime
from sqlalchemy import Text

class TextPreprocessing(db.Model):
    __tablename__ = 'text_preprocessing'

    id = db.Column(db.Integer, primary_key=True)
    tweet_id = db.Column(db.Integer, db.ForeignKey('twitter_scraping.id'), nullable=False)
    
    original_text = db.Column(Text, nullable=False)
    cleaned_text = db.Column(Text, nullable=True)
    case_folded_text = db.Column(Text, nullable=True)
    tokenized_text = db.Column(Text, nullable=True)
    filtered_text = db.Column(Text, nullable=True)
    normalized_text = db.Column(Text, nullable=True)
    stemmed_text = db.Column(Text, nullable=True)
    final_text = db.Column(Text, nullable=True)
    
    processing_status = db.Column(db.Enum('pending', 'processing', 'completed', 'failed'), default='pending')
    processed_at = db.Column(db.DateTime, nullable=True)
    processed_by = db.Column(db.Integer, db.ForeignKey('users.id_user'), nullable=False)
    
    word_count_before = db.Column(db.Integer, default=0)
    word_count_after = db.Column(db.Integer, default=0)
    removed_urls = db.Column(db.Integer, default=0)
    removed_mentions = db.Column(db.Integer, default=0)
    removed_hashtags = db.Column(db.Integer, default=0)
    removed_stopwords = db.Column(db.Integer, default=0)
    normalized_words = db.Column(db.Integer, default=0)
    stemmed_words = db.Column(db.Integer, default=0)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<TextPreprocessing {self.id}>'

class PreprocessingSettings(db.Model):
    __tablename__ = 'preprocessing_settings'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id_user'), nullable=False)
    
    enable_cleansing = db.Column(db.Boolean, default=True)
    enable_case_folding = db.Column(db.Boolean, default=True)
    enable_tokenizing = db.Column(db.Boolean, default=True)
    enable_stopword_removal = db.Column(db.Boolean, default=True)
    enable_normalization = db.Column(db.Boolean, default=True)
    enable_stemming = db.Column(db.Boolean, default=True)
    
    remove_urls = db.Column(db.Boolean, default=True)
    remove_mentions = db.Column(db.Boolean, default=True)
    remove_hashtags = db.Column(db.Boolean, default=False)
    remove_numbers = db.Column(db.Boolean, default=True)
    remove_punctuation = db.Column(db.Boolean, default=True)
    remove_emoticons = db.Column(db.Boolean, default=False)
    
    min_word_length = db.Column(db.Integer, default=2)
    max_word_length = db.Column(db.Integer, default=50)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<PreprocessingSettings {self.user_id}>'

class StopwordList(db.Model):
    __tablename__ = 'stopword_list'
    
    id = db.Column(db.Integer, primary_key=True)
    word = db.Column(db.String(50), unique=True, nullable=False)
    category = db.Column(db.String(50), default='general')
    is_active = db.Column(db.Boolean, default=True)
    added_by = db.Column(db.Integer, db.ForeignKey('users.id_user'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<StopwordList {self.word}>'

class NormalizationDict(db.Model):
    __tablename__ = 'normalization_dict'
    
    id = db.Column(db.Integer, primary_key=True)
    slang_word = db.Column(db.String(100), unique=True, nullable=False)
    standard_word = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(50), default='general')
    is_active = db.Column(db.Boolean, default=True)
    usage_count = db.Column(db.Integer, default=0)
    added_by = db.Column(db.Integer, db.ForeignKey('users.id_user'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<NormalizationDict {self.slang_word} -> {self.standard_word}>'