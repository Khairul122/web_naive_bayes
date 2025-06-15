from flask import Blueprint, render_template, request, redirect, url_for, flash, session, send_file
from app.models.SentimenModel import SentimentAnalysis
from app.models.ScrappingModel import TwitterScraping
from app.models.PreprocessingModel import TextPreprocessing
from app.extension import db
from datetime import datetime
import pandas as pd
import os
import re

sentimen_bp = Blueprint('sentimen', __name__, url_prefix='/sentimen')

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Silakan login terlebih dahulu', 'warning')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function

@sentimen_bp.route('/')
@login_required
def index():
    user_id = session['user_id']
    
    total_tweets = TwitterScraping.query.filter_by(scraped_by=user_id).count()
    labeled_count = SentimentAnalysis.query.filter_by(labeled_by=user_id).count()
    pending_count = total_tweets - labeled_count
    
    positif_count = SentimentAnalysis.query.filter_by(labeled_by=user_id, sentiment_label='positif').count()
    negatif_count = SentimentAnalysis.query.filter_by(labeled_by=user_id, sentiment_label='negatif').count()
    netral_count = SentimentAnalysis.query.filter_by(labeled_by=user_id, sentiment_label='netral').count()
    
    page = request.args.get('page', 1, type=int)
    sentiment_data = db.session.query(SentimentAnalysis, TwitterScraping)\
        .join(TwitterScraping, SentimentAnalysis.tweet_id == TwitterScraping.id)\
        .filter(SentimentAnalysis.labeled_by == user_id)\
        .order_by(SentimentAnalysis.created_at.desc())\
        .paginate(page=page, per_page=20, error_out=False)
    
    stats = {
        'total_tweets': total_tweets,
        'labeled_count': labeled_count,
        'pending_count': pending_count,
        'positif_count': positif_count,
        'negatif_count': negatif_count,
        'netral_count': netral_count,
        'labeling_rate': round((labeled_count / total_tweets * 100), 2) if total_tweets > 0 else 0
    }
    
    return render_template('sentimen/index.html', sentiment_data=sentiment_data, stats=stats)

@sentimen_bp.route('/auto_label', methods=['POST'])
@login_required
def auto_label():
    user_id = session['user_id']
    
    unlabeled_tweets = db.session.query(TwitterScraping)\
        .outerjoin(SentimentAnalysis, TwitterScraping.id == SentimentAnalysis.tweet_id)\
        .filter(TwitterScraping.scraped_by == user_id)\
        .filter(SentimentAnalysis.id.is_(None))\
        .all()
    
    if not unlabeled_tweets:
        flash('Semua tweet sudah dilabeling', 'info')
        return redirect(url_for('sentimen.index'))
    
    labeled_count = 0
    for tweet in unlabeled_tweets:
        try:
            preprocessing = TextPreprocessing.query.filter_by(tweet_id=tweet.id, processed_by=user_id).first()
            text_to_analyze = preprocessing.final_text if preprocessing and preprocessing.final_text else tweet.full_text
            
            sentiment_result = auto_label_sentiment(text_to_analyze)
            
            sentiment_record = SentimentAnalysis(
                tweet_id=tweet.id,
                preprocessing_id=preprocessing.id if preprocessing else None,
                username=tweet.username,
                tweet_text=tweet.full_text,
                processed_text=text_to_analyze,
                sentiment_label=sentiment_result['label'],
                confidence_score=sentiment_result['confidence'],
                positive_keywords=', '.join(sentiment_result['positive_keywords']),
                negative_keywords=', '.join(sentiment_result['negative_keywords']),
                neutral_keywords=', '.join(sentiment_result['neutral_keywords']),
                labeling_method='auto',
                labeled_at=datetime.utcnow(),
                labeled_by=user_id
            )
            
            db.session.add(sentiment_record)
            labeled_count += 1
            
        except Exception as e:
            continue
    
    db.session.commit()
    flash(f'Berhasil melabeling {labeled_count} tweet secara otomatis', 'success')
    return redirect(url_for('sentimen.index'))

@sentimen_bp.route('/reset', methods=['POST'])
@login_required
def reset():
    user_id = session['user_id']
    SentimentAnalysis.query.filter_by(labeled_by=user_id).delete()
    db.session.commit()
    flash('Semua data sentiment berhasil dihapus', 'success')
    return redirect(url_for('sentimen.index'))

@sentimen_bp.route('/export')
@login_required
def export():
    user_id = session['user_id']
    
    results = db.session.query(SentimentAnalysis)\
        .filter(SentimentAnalysis.labeled_by == user_id)\
        .all()
    
    if not results:
        flash('Tidak ada data untuk diexport', 'warning')
        return redirect(url_for('sentimen.index'))
    
    data = []
    for sentiment in results:
        data.append({
            'username': sentiment.username,
            'tweet': sentiment.tweet_text,
            'label': sentiment.sentiment_label,
            'confidence': sentiment.confidence_score,
            'method': sentiment.labeling_method,
            'labeled_at': sentiment.labeled_at
        })
    
    df = pd.DataFrame(data)
    
    from flask import current_app
    dataset_dir = os.path.join(current_app.root_path, 'static', 'dataset')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    filename = f"sentiment_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    export_path = os.path.join(dataset_dir, filename)
    df.to_csv(export_path, index=False)
    
    return send_file(export_path, as_attachment=True, download_name=filename, mimetype='text/csv')

def auto_label_sentiment(text):
    positive_keywords = [
        'bagus', 'baik', 'hebat', 'mantap', 'keren', 'sukses', 'berhasil', 'positif',
        'senang', 'gembira', 'bangga', 'puas', 'suka', 'cinta', 'sayang', 'terima kasih',
        'luar biasa', 'fantastis', 'menawan', 'indah', 'cantik', 'tampan', 'pintar',
        'bijak', 'benar', 'setuju', 'mendukung', 'optimis', 'harapan', 'semangat',
        'maju', 'berkembang', 'meningkat', 'naik', 'bertambah', 'membaik'
    ]
    
    negative_keywords = [
        'buruk', 'jelek', 'bodoh', 'tolol', 'benci', 'marah', 'kesal', 'sedih',
        'kecewa', 'gagal', 'rusak', 'hancur', 'parah', 'gila', 'stress', 'capek',
        'lelah', 'sakit', 'mati', 'maut', 'korupsi', 'bohong', 'tipu', 'curang',
        'salah', 'tidak', 'jangan', 'bukan', 'menolak', 'protes', 'demo', 'kritik',
        'turun', 'menurun', 'berkurang', 'memburuk', 'rusuh', 'chaos', 'krisis'
    ]
    
    neutral_keywords = [
        'mungkin', 'sepertinya', 'kayaknya', 'agak', 'cukup', 'lumayan', 'standar',
        'biasa', 'normal', 'wajar', 'begitu', 'gitu', 'aja', 'saja', 'hanya',
        'akan', 'sedang', 'lagi', 'sudah', 'telah', 'pernah', 'belum', 'masih'
    ]
    
    text_lower = text.lower()
    
    positive_found = []
    negative_found = []
    neutral_found = []
    
    for keyword in positive_keywords:
        if re.search(r'\b' + keyword + r'\b', text_lower):
            positive_found.append(keyword)
    
    for keyword in negative_keywords:
        if re.search(r'\b' + keyword + r'\b', text_lower):
            negative_found.append(keyword)
    
    for keyword in neutral_keywords:
        if re.search(r'\b' + keyword + r'\b', text_lower):
            neutral_found.append(keyword)
    
    positive_score = len(positive_found)
    negative_score = len(negative_found)
    neutral_score = len(neutral_found)
    
    total_score = positive_score + negative_score + neutral_score
    
    if total_score == 0:
        label = 'netral'
        confidence = 0.5
    elif positive_score > negative_score and positive_score > neutral_score:
        label = 'positif'
        confidence = round(positive_score / total_score, 2)
    elif negative_score > positive_score and negative_score > neutral_score:
        label = 'negatif'
        confidence = round(negative_score / total_score, 2)
    else:
        label = 'netral'
        confidence = round(neutral_score / total_score if neutral_score > 0 else 0.5, 2)
    
    return {
        'label': label,
        'confidence': confidence,
        'positive_keywords': positive_found,
        'negative_keywords': negative_found,
        'neutral_keywords': neutral_found
    }