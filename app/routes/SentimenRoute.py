from flask import Blueprint, render_template, request, redirect, url_for, flash, session, send_file, current_app
from app.models.SentimenModel import SentimentAnalysis, SentimentSettings
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

    settings = SentimentSettings.query.filter_by(user_id=user_id).first()

    try:
        pos_dict, neg_dict = load_inset_lexicon()
    except Exception:
        pos_dict, neg_dict = {}, {}

    item_scores = {}
    for sentiment, tweet in sentiment_data.items:
        pos_words = [w.strip() for w in (sentiment.positive_keywords or '').split(',') if w.strip()]
        neg_words = [w.strip() for w in (sentiment.negative_keywords or '').split(',') if w.strip()]
        score = sum(pos_dict.get(w, 0) for w in pos_words) + sum(neg_dict.get(w, 0) for w in neg_words)
        all_words = pos_words + neg_words
        item_scores[sentiment.id] = {
            'inset_score': round(score, 2),
            'matched_words': ', '.join(all_words) if all_words else '—'
        }

    return render_template(
        'sentimen/index.html',
        sentiment_data=sentiment_data,
        stats=stats,
        settings=settings,
        item_scores=item_scores
    )

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
    
    settings = SentimentSettings.query.filter_by(user_id=user_id).first()
    threshold_pos = settings.threshold_pos if settings else 0.5
    threshold_neg = settings.threshold_neg if settings else -0.5
    use_keyword_engine = settings.labeling_engine == 'keyword' if settings else False

    try:
        pos_dict, neg_dict = load_inset_lexicon()
    except Exception:
        pos_dict, neg_dict = None, None

    labeled_count = 0
    for tweet in unlabeled_tweets:
        try:
            preprocessing = TextPreprocessing.query.filter_by(tweet_id=tweet.id, processed_by=user_id).first()
            text_to_analyze = preprocessing.final_text if preprocessing and preprocessing.final_text else tweet.full_text

            if use_keyword_engine or pos_dict is None:
                sentiment_result = _fallback_keyword_label(text_to_analyze)
            else:
                sentiment_result = inset_label_sentiment(text_to_analyze, pos_dict, neg_dict, threshold_pos, threshold_neg)

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

@sentimen_bp.route('/save_settings', methods=['POST'])
@login_required
def save_settings():
    user_id = session['user_id']
    threshold_pos = float(request.form.get('threshold_pos', 0.5))
    threshold_neg = float(request.form.get('threshold_neg', -0.5))
    engine = request.form.get('labeling_engine', 'inset')

    if threshold_pos <= 0:
        flash('Threshold positif harus lebih besar dari 0', 'warning')
        return redirect(url_for('sentimen.index'))
    if threshold_neg >= 0:
        flash('Threshold negatif harus lebih kecil dari 0', 'warning')
        return redirect(url_for('sentimen.index'))

    settings = SentimentSettings.query.filter_by(user_id=user_id).first()
    if not settings:
        settings = SentimentSettings(user_id=user_id)
        db.session.add(settings)

    settings.threshold_pos = threshold_pos
    settings.threshold_neg = threshold_neg
    settings.labeling_engine = engine
    db.session.commit()

    flash('Pengaturan labeling berhasil disimpan', 'success')
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

    dataset_dir = os.path.join(current_app.root_path, 'static', 'dataset')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    filename = f"sentiment_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    export_path = os.path.join(dataset_dir, filename)
    df.to_csv(export_path, index=False)
    
    return send_file(export_path, as_attachment=True, download_name=filename, mimetype='text/csv')

def load_inset_lexicon():
    """Load InSet Lexicon CSV files into dicts, cached in current_app.config."""
    if 'INSET_POS' not in current_app.config:
        base = os.path.join(os.path.dirname(current_app.root_path), 'database')
        pos_df = pd.read_csv(os.path.join(base, 'inset_lexicon_positive.csv'))
        neg_df = pd.read_csv(os.path.join(base, 'inset_lexicon_negative.csv'))
        current_app.config['INSET_POS'] = dict(zip(pos_df['word'], pos_df['weight']))
        current_app.config['INSET_NEG'] = dict(zip(neg_df['word'], neg_df['weight']))

    return current_app.config['INSET_POS'], current_app.config['INSET_NEG']


def inset_label_sentiment(text, pos_dict, neg_dict, threshold_pos=0.5, threshold_neg=-0.5):
    """Label sentiment using InSet Lexicon weighted word scoring."""
    tokens = text.lower().split()

    positive_found = []
    negative_found = []

    for token in tokens:
        if token in pos_dict:
            positive_found.append(token)
        if token in neg_dict:
            negative_found.append(token)

    pos_score = sum(pos_dict[w] for w in positive_found)
    neg_score = sum(neg_dict[w] for w in negative_found)
    total_score = pos_score + neg_score

    abs_score = abs(total_score)
    confidence = min(round(abs_score / 5.0, 2), 1.0)

    if total_score > threshold_pos:
        label = 'positif'
    elif total_score < threshold_neg:
        label = 'negatif'
    else:
        label = 'netral'

    return {
        'label': label,
        'confidence': confidence,
        'positive_keywords': positive_found,
        'negative_keywords': negative_found,
        'neutral_keywords': [],
    }


def _fallback_keyword_label(text):
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