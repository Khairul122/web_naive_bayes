from flask import Blueprint, render_template, request, redirect, url_for, flash, session, send_file
from app.models.KonversiModel import TfidfConversion, TfidfVocabulary
from app.models.SentimenModel import SentimentAnalysis
from app.extension import db
from datetime import datetime
import pandas as pd
import numpy as np
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

konversi_bp = Blueprint('konversi', __name__, url_prefix='/konversi')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Silakan login terlebih dahulu', 'warning')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function

@konversi_bp.route('/')
@login_required
def index():
    user_id = session['user_id']
    
    total_sentiments = SentimentAnalysis.query.filter_by(labeled_by=user_id).count()
    converted_count = TfidfConversion.query.filter_by(converted_by=user_id).count()
    pending_count = total_sentiments - converted_count
    
    total_features = db.session.query(db.func.max(TfidfConversion.total_features))\
        .filter_by(converted_by=user_id).scalar() or 0
    
    vocabulary_count = TfidfVocabulary.query.filter_by(user_id=user_id).count()
    
    page = request.args.get('page', 1, type=int)
    conversion_data = db.session.query(TfidfConversion, SentimentAnalysis)\
        .join(SentimentAnalysis, TfidfConversion.sentiment_id == SentimentAnalysis.id)\
        .filter(TfidfConversion.converted_by == user_id)\
        .order_by(TfidfConversion.created_at.desc())\
        .paginate(page=page, per_page=20, error_out=False)
    
    stats = {
        'total_sentiments': total_sentiments,
        'converted_count': converted_count,
        'pending_count': pending_count,
        'total_features': total_features,
        'vocabulary_count': vocabulary_count,
        'conversion_rate': round((converted_count / total_sentiments * 100), 2) if total_sentiments > 0 else 0
    }
    
    return render_template('konversi/index.html', conversion_data=conversion_data, stats=stats)

@konversi_bp.route('/convert', methods=['POST'])
@login_required
def convert_tfidf():
    user_id = session['user_id']
    
    max_features = int(request.form.get('max_features', 1000))
    min_df = float(request.form.get('min_df', 0.01))
    max_df = float(request.form.get('max_df', 0.95))
    
    unconverted_sentiments = db.session.query(SentimentAnalysis)\
        .outerjoin(TfidfConversion, SentimentAnalysis.id == TfidfConversion.sentiment_id)\
        .filter(SentimentAnalysis.labeled_by == user_id)\
        .filter(TfidfConversion.id.is_(None))\
        .all()
    
    if not unconverted_sentiments:
        flash('Semua data sentiment sudah dikonversi', 'info')
        return redirect(url_for('konversi.index'))
    
    if len(unconverted_sentiments) < 2:
        flash('Minimal 2 dokumen diperlukan untuk TF-IDF', 'warning')
        return redirect(url_for('konversi.index'))
    
    try:
        texts = []
        sentiment_ids = []
        
        for sentiment in unconverted_sentiments:
            text = sentiment.processed_text if sentiment.processed_text else sentiment.tweet_text
            if text and text.strip():
                texts.append(text.strip())
                sentiment_ids.append(sentiment.id)
        
        if len(texts) < 2:
            flash('Tidak cukup teks yang valid untuk TF-IDF', 'warning')
            return redirect(url_for('konversi.index'))
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words=None,
            lowercase=True,
            token_pattern=r'\b\w+\b'
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        TfidfVocabulary.query.filter_by(user_id=user_id).delete()
        
        for idx, term in enumerate(feature_names):
            vocab = TfidfVocabulary(
                user_id=user_id,
                term=term,
                feature_index=idx,
                document_frequency=vectorizer.vocabulary_.get(term, 0),
                idf_score=vectorizer.idf_[idx]
            )
            db.session.add(vocab)
        
        converted_count = 0
        for i, sentiment_id in enumerate(sentiment_ids):
            try:
                feature_vector = tfidf_matrix[i].toarray().flatten()
                
                conversion_record = TfidfConversion(
                    sentiment_id=sentiment_id,
                    text_input=texts[i],
                    feature_vector=json.dumps(feature_vector.tolist()),
                    feature_names=json.dumps(feature_names.tolist()),
                    total_features=len(feature_names),
                    max_features=max_features,
                    min_df=min_df,
                    max_df=max_df,
                    conversion_method='tfidf',
                    converted_by=user_id
                )
                
                db.session.add(conversion_record)
                converted_count += 1
                
            except Exception as e:
                logger.error(f"Error converting sentiment {sentiment_id}: {e}")
                continue
        
        db.session.commit()
        flash(f'Berhasil konversi TF-IDF untuk {converted_count} data dengan {len(feature_names)} fitur', 'success')
        
    except Exception as e:
        logger.error(f"TF-IDF conversion error: {e}")
        flash(f'Error konversi TF-IDF: {str(e)}', 'danger')
    
    return redirect(url_for('konversi.index'))

@konversi_bp.route('/vocabulary')
@login_required
def vocabulary():
    user_id = session['user_id']
    
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '')
    
    query = TfidfVocabulary.query.filter_by(user_id=user_id)
    
    if search:
        query = query.filter(TfidfVocabulary.term.contains(search))
    
    vocab_data = query.order_by(TfidfVocabulary.idf_score.desc())\
        .paginate(page=page, per_page=50, error_out=False)
    
    return render_template('konversi/vocabulary.html', vocab_data=vocab_data, search=search)

@konversi_bp.route('/reset', methods=['POST'])
@login_required
def reset():
    user_id = session['user_id']
    TfidfConversion.query.filter_by(converted_by=user_id).delete()
    TfidfVocabulary.query.filter_by(user_id=user_id).delete()
    db.session.commit()
    flash('Semua data TF-IDF berhasil dihapus', 'success')
    return redirect(url_for('konversi.index'))

@konversi_bp.route('/export')
@login_required
def export():
    user_id = session['user_id']
    
    results = db.session.query(TfidfConversion, SentimentAnalysis)\
        .join(SentimentAnalysis, TfidfConversion.sentiment_id == SentimentAnalysis.id)\
        .filter(TfidfConversion.converted_by == user_id)\
        .all()
    
    if not results:
        flash('Tidak ada data untuk diexport', 'warning')
        return redirect(url_for('konversi.index'))
    
    data = []
    for conversion, sentiment in results:
        feature_vector = json.loads(conversion.feature_vector)
        feature_names = json.loads(conversion.feature_names)
        
        row = {
            'username': sentiment.username,
            'tweet': sentiment.tweet_text,
            'sentiment_label': sentiment.sentiment_label,
            'total_features': conversion.total_features,
            'max_tfidf_score': max(feature_vector) if feature_vector else 0,
            'non_zero_features': sum(1 for x in feature_vector if x > 0)
        }
        
        for i, feature_name in enumerate(feature_names[:10]):
            row[f'feature_{feature_name}'] = feature_vector[i] if i < len(feature_vector) else 0
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    from flask import current_app
    dataset_dir = os.path.join(current_app.root_path, 'static', 'dataset')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    filename = f"tfidf_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    export_path = os.path.join(dataset_dir, filename)
    df.to_csv(export_path, index=False)
    
    return send_file(export_path, as_attachment=True, download_name=filename, mimetype='text/csv')

@konversi_bp.route('/export_matrix')
@login_required
def export_matrix():
    user_id = session['user_id']
    
    results = db.session.query(TfidfConversion, SentimentAnalysis)\
        .join(SentimentAnalysis, TfidfConversion.sentiment_id == SentimentAnalysis.id)\
        .filter(TfidfConversion.converted_by == user_id)\
        .all()
    
    if not results:
        flash('Tidak ada data untuk diexport', 'warning')
        return redirect(url_for('konversi.index'))
    
    matrix_data = []
    feature_names = None
    
    for conversion, sentiment in results:
        feature_vector = json.loads(conversion.feature_vector)
        if feature_names is None:
            feature_names = json.loads(conversion.feature_names)
        
        row = [sentiment.username, sentiment.tweet_text, sentiment.sentiment_label] + feature_vector
        matrix_data.append(row)
    
    columns = ['username', 'tweet', 'label'] + [f'feature_{name}' for name in feature_names]
    df = pd.DataFrame(matrix_data, columns=columns)
    
    from flask import current_app
    dataset_dir = os.path.join(current_app.root_path, 'static', 'dataset')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    filename = f"tfidf_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    export_path = os.path.join(dataset_dir, filename)
    df.to_csv(export_path, index=False)
    
    return send_file(export_path, as_attachment=True, download_name=filename, mimetype='text/csv')