from app.extension import db
from datetime import datetime

class NBCTraining(db.Model):
    __tablename__ = 'nbc_training'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id_user'), nullable=False)
    conversion_id = db.Column(db.Integer, db.ForeignKey('tfidf_conversion.id'), nullable=False)
    
    feature_vector = db.Column(db.Text, nullable=False)
    label = db.Column(db.String(20), nullable=False)
    
    split_method = db.Column(db.String(50), default='train_test_split')
    test_size = db.Column(db.Float, default=0.3)
    random_state = db.Column(db.Integer, default=42)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<NBCTraining {self.id}>'

class NBCTesting(db.Model):
    __tablename__ = 'nbc_testing'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id_user'), nullable=False)
    conversion_id = db.Column(db.Integer, db.ForeignKey('tfidf_conversion.id'), nullable=False)
    
    feature_vector = db.Column(db.Text, nullable=False)
    true_label = db.Column(db.String(20), nullable=False)
    predicted_label = db.Column(db.String(20), nullable=True)
    prediction_probability = db.Column(db.Text, nullable=True)
    is_correct = db.Column(db.Boolean, default=False)
    
    split_method = db.Column(db.String(50), default='train_test_split')
    test_size = db.Column(db.Float, default=0.3)
    random_state = db.Column(db.Integer, default=42)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<NBCTesting {self.id}>'

class NBCModel(db.Model):
    __tablename__ = 'nbc_model'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id_user'), nullable=False)
    
    model_name = db.Column(db.String(100), nullable=False)
    alpha = db.Column(db.Float, default=1.0)
    
    feature_log_prob = db.Column(db.Text, nullable=False)
    class_log_prior = db.Column(db.Text, nullable=False)
    classes = db.Column(db.Text, nullable=False)
    
    n_features = db.Column(db.Integer, default=0)
    n_classes = db.Column(db.Integer, default=0)
    training_samples = db.Column(db.Integer, default=0)
    testing_samples = db.Column(db.Integer, default=0)
    
    accuracy = db.Column(db.Float, default=0.0)
    training_method = db.Column(db.String(50), default='MultinomialNB')
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    trained_at = db.Column(db.DateTime, default=datetime.utcnow)
    tested_at = db.Column(db.DateTime, nullable=True)
    
    def __repr__(self):
        return f'<NBCModel {self.model_name}>'