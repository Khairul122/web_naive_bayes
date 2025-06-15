from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from app.models.PreprocessingModel import TextPreprocessing, PreprocessingSettings, StopwordList, NormalizationDict
from app.models.ScrappingModel import TwitterScraping
from app.extension import db
from datetime import datetime
import re
import string
import pandas as pd
import logging

preprocessing_bp = Blueprint('preprocessing', __name__, url_prefix='/preprocessing')

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

@preprocessing_bp.route('/')
@login_required
def index():
    user_id = session['user_id']
    
    total_tweets = TwitterScraping.query.filter_by(scraped_by=user_id).count()
    processed_count = TextPreprocessing.query.filter_by(processed_by=user_id, processing_status='completed').count()
    pending_count = total_tweets - processed_count
    
    page = request.args.get('page', 1, type=int)
    preprocessing_data = db.session.query(TextPreprocessing, TwitterScraping)\
        .join(TwitterScraping, TextPreprocessing.tweet_id == TwitterScraping.id)\
        .filter(TextPreprocessing.processed_by == user_id)\
        .order_by(TextPreprocessing.created_at.desc())\
        .paginate(page=page, per_page=20, error_out=False)
    
    settings = PreprocessingSettings.query.filter_by(user_id=user_id).first()
    if not settings:
        settings = PreprocessingSettings(user_id=user_id)
        db.session.add(settings)
        db.session.commit()
    
    stats = {
        'total_tweets': total_tweets,
        'processed_count': processed_count,
        'pending_count': pending_count,
        'processing_rate': round((processed_count / total_tweets * 100), 2) if total_tweets > 0 else 0
    }
    
    return render_template('preprocessing/index.html', 
                         preprocessing_data=preprocessing_data,
                         settings=settings,
                         stats=stats)

@preprocessing_bp.route('/process', methods=['POST'])
@login_required
def process_texts():
    user_id = session['user_id']
    
    unprocessed_tweets = TwitterScraping.query.filter_by(scraped_by=user_id)\
        .filter(~TwitterScraping.id.in_(
            db.session.query(TextPreprocessing.tweet_id)
            .filter_by(processed_by=user_id)
        )).all()
    
    if not unprocessed_tweets:
        flash('Semua tweet sudah diproses', 'info')
        return redirect(url_for('preprocessing.index'))
    
    settings = PreprocessingSettings.query.filter_by(user_id=user_id).first()
    
    processed_count = 0
    for tweet in unprocessed_tweets:
        try:
            preprocessing_record = TextPreprocessing(
                tweet_id=tweet.id,
                original_text=tweet.full_text,
                processed_by=user_id,
                processing_status='processing'
            )
            db.session.add(preprocessing_record)
            db.session.flush()
            
            result = preprocess_text(tweet.full_text, settings)
            
            preprocessing_record.cleaned_text = result['cleaned_text']
            preprocessing_record.case_folded_text = result['case_folded_text']
            preprocessing_record.tokenized_text = result['tokenized_text']
            preprocessing_record.filtered_text = result['filtered_text']
            preprocessing_record.normalized_text = result['normalized_text']
            preprocessing_record.stemmed_text = result['stemmed_text']
            preprocessing_record.final_text = result['final_text']
            
            preprocessing_record.word_count_before = result['stats']['word_count_before']
            preprocessing_record.word_count_after = result['stats']['word_count_after']
            preprocessing_record.removed_urls = result['stats']['removed_urls']
            preprocessing_record.removed_mentions = result['stats']['removed_mentions']
            preprocessing_record.removed_hashtags = result['stats']['removed_hashtags']
            preprocessing_record.removed_stopwords = result['stats']['removed_stopwords']
            preprocessing_record.normalized_words = result['stats']['normalized_words']
            preprocessing_record.stemmed_words = result['stats']['stemmed_words']
            
            preprocessing_record.processing_status = 'completed'
            preprocessing_record.processed_at = datetime.utcnow()
            
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing tweet {tweet.id}: {e}")
            preprocessing_record.processing_status = 'failed'
    
    db.session.commit()
    flash(f'Berhasil memproses {processed_count} tweet', 'success')
    return redirect(url_for('preprocessing.index'))

@preprocessing_bp.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    user_id = session['user_id']
    settings = PreprocessingSettings.query.filter_by(user_id=user_id).first()
    
    if not settings:
        settings = PreprocessingSettings(user_id=user_id)
        db.session.add(settings)
        db.session.commit()
    
    if request.method == 'POST':
        settings.enable_cleansing = bool(request.form.get('enable_cleansing'))
        settings.enable_case_folding = bool(request.form.get('enable_case_folding'))
        settings.enable_tokenizing = bool(request.form.get('enable_tokenizing'))
        settings.enable_stopword_removal = bool(request.form.get('enable_stopword_removal'))
        settings.enable_normalization = bool(request.form.get('enable_normalization'))
        settings.enable_stemming = bool(request.form.get('enable_stemming'))
        
        settings.remove_urls = bool(request.form.get('remove_urls'))
        settings.remove_mentions = bool(request.form.get('remove_mentions'))
        settings.remove_hashtags = bool(request.form.get('remove_hashtags'))
        settings.remove_numbers = bool(request.form.get('remove_numbers'))
        settings.remove_punctuation = bool(request.form.get('remove_punctuation'))
        settings.remove_emoticons = bool(request.form.get('remove_emoticons'))
        
        settings.min_word_length = int(request.form.get('min_word_length', 2))
        settings.max_word_length = int(request.form.get('max_word_length', 50))
        settings.updated_at = datetime.utcnow()
        
        db.session.commit()
        flash('Pengaturan berhasil disimpan', 'success')
        return redirect(url_for('preprocessing.settings'))
    
    stopwords = StopwordList.query.filter_by(is_active=True).all()
    normalizations = NormalizationDict.query.filter_by(is_active=True).all()
    
    return render_template('preprocessing/settings.html', 
                         settings=settings,
                         stopwords=stopwords,
                         normalizations=normalizations)

@preprocessing_bp.route('/reset', methods=['POST'])
@login_required
def reset():
    user_id = session['user_id']
    TextPreprocessing.query.filter_by(processed_by=user_id).delete()
    db.session.commit()
    flash('Semua data preprocessing berhasil dihapus', 'success')
    return redirect(url_for('preprocessing.index'))

@preprocessing_bp.route('/export')
@login_required
def export():
    user_id = session['user_id']
    
    results = db.session.query(TextPreprocessing, TwitterScraping)\
        .join(TwitterScraping, TextPreprocessing.tweet_id == TwitterScraping.id)\
        .filter(TextPreprocessing.processed_by == user_id)\
        .all()
    
    if not results:
        flash('Tidak ada data untuk diexport', 'warning')
        return redirect(url_for('preprocessing.index'))
    
    data = []
    for preprocessing, tweet in results:
        data.append({
            'tweet_id': tweet.tweet_id_str,
            'username': tweet.username,
            'original_text': preprocessing.original_text,
            'final_text': preprocessing.final_text,
            'word_count_before': preprocessing.word_count_before,
            'word_count_after': preprocessing.word_count_after,
            'removed_urls': preprocessing.removed_urls,
            'removed_mentions': preprocessing.removed_mentions,
            'removed_hashtags': preprocessing.removed_hashtags,
            'removed_stopwords': preprocessing.removed_stopwords,
            'normalized_words': preprocessing.normalized_words,
            'stemmed_words': preprocessing.stemmed_words,
            'processed_at': preprocessing.processed_at
        })
    
    df = pd.DataFrame(data)
    
    from flask import current_app
    import os
    dataset_dir = os.path.join(current_app.root_path, 'static', 'dataset')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    filename = f"preprocessing_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    export_path = os.path.join(dataset_dir, filename)
    df.to_csv(export_path, index=False)
    
    from flask import send_file
    return send_file(export_path, as_attachment=True, download_name=filename, mimetype='text/csv')

def preprocess_text(text, settings):
    stats = {
        'word_count_before': len(text.split()),
        'removed_urls': 0,
        'removed_mentions': 0,
        'removed_hashtags': 0,
        'removed_stopwords': 0,
        'normalized_words': 0,
        'stemmed_words': 0,
        'word_count_after': 0
    }
    
    original_text = text
    processed_text = text
    
    if settings.enable_cleansing:
        processed_text = cleansing_text(processed_text, settings, stats)
    
    if settings.enable_case_folding:
        processed_text = case_folding(processed_text)
    
    if settings.enable_tokenizing:
        tokens = tokenizing(processed_text)
        tokenized_text = ' '.join(tokens)
    else:
        tokens = processed_text.split()
        tokenized_text = processed_text
    
    if settings.enable_stopword_removal:
        tokens = stopword_removal(tokens, stats)
        filtered_text = ' '.join(tokens)
    else:
        filtered_text = tokenized_text
    
    if settings.enable_normalization:
        tokens = normalization(tokens, stats)
        normalized_text = ' '.join(tokens)
    else:
        normalized_text = filtered_text
    
    if settings.enable_stemming:
        tokens = stemming_ecs(tokens, stats)
        stemmed_text = ' '.join(tokens)
    else:
        stemmed_text = normalized_text
    
    final_tokens = [token for token in tokens if settings.min_word_length <= len(token) <= settings.max_word_length]
    final_text = ' '.join(final_tokens)
    
    stats['word_count_after'] = len(final_tokens)
    
    return {
        'original_text': original_text,
        'cleaned_text': processed_text if settings.enable_cleansing else original_text,
        'case_folded_text': case_folding(processed_text) if settings.enable_case_folding else processed_text,
        'tokenized_text': tokenized_text,
        'filtered_text': filtered_text,
        'normalized_text': normalized_text,
        'stemmed_text': stemmed_text,
        'final_text': final_text,
        'stats': stats
    }

def cleansing_text(text, settings, stats):
    if settings.remove_urls:
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        stats['removed_urls'] = len(re.findall(url_pattern, text))
        text = re.sub(url_pattern, '', text)
    
    if settings.remove_mentions:
        mention_pattern = r'@\w+'
        stats['removed_mentions'] = len(re.findall(mention_pattern, text))
        text = re.sub(mention_pattern, '', text)
    
    if settings.remove_hashtags:
        hashtag_pattern = r'#\w+'
        stats['removed_hashtags'] = len(re.findall(hashtag_pattern, text))
        text = re.sub(hashtag_pattern, '', text)
    
    if settings.remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    if settings.remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    if settings.remove_emoticons:
        emoticon_pattern = r'[^\w\s]'
        text = re.sub(emoticon_pattern, '', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def case_folding(text):
    return text.lower()

def tokenizing(text):
    tokens = text.split()
    return [token for token in tokens if token.strip()]

def stopword_removal(tokens, stats):
    stopwords = {word.word for word in StopwordList.query.filter_by(is_active=True).all()}
    
    if not stopwords:
        default_stopwords = {
            'dan', 'atau', 'yang', 'di', 'ke', 'dari', 'untuk', 'dengan', 'pada', 'dalam',
            'adalah', 'akan', 'telah', 'sudah', 'juga', 'tidak', 'bukan', 'tetapi', 'namun',
            'karena', 'sebab', 'oleh', 'agar', 'supaya', 'jika', 'kalau', 'bila', 'ketika',
            'saat', 'waktu', 'ini', 'itu', 'tersebut', 'maka', 'lalu', 'kemudian', 'setelah'
        }
        stopwords = default_stopwords
    
    original_count = len(tokens)
    filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    stats['removed_stopwords'] = original_count - len(filtered_tokens)
    
    return filtered_tokens

def normalization(tokens, stats):
    normalization_dict = {norm.slang_word: norm.standard_word 
                         for norm in NormalizationDict.query.filter_by(is_active=True).all()}
    
    if not normalization_dict:
        default_normalization = {
            'gak': 'tidak', 'ga': 'tidak', 'nggak': 'tidak', 'enggak': 'tidak',
            'udah': 'sudah', 'dah': 'sudah', 'udh': 'sudah',
            'emang': 'memang', 'mang': 'memang',
            'gimana': 'bagaimana', 'gmn': 'bagaimana',
            'kenapa': 'mengapa', 'knp': 'mengapa',
            'dimana': 'di mana', 'dmn': 'di mana',
            'bgt': 'banget', 'bener': 'benar', 'bnr': 'benar',
            'skrg': 'sekarang', 'krn': 'karena', 'krna': 'karena',
            'lg': 'lagi', 'lgi': 'lagi', 'trs': 'terus',
            'yg': 'yang', 'dgn': 'dengan', 'dr': 'dari'
        }
        normalization_dict = default_normalization
    
    normalized_tokens = []
    for token in tokens:
        if token.lower() in normalization_dict:
            normalized_tokens.append(normalization_dict[token.lower()])
            stats['normalized_words'] += 1
        else:
            normalized_tokens.append(token)
    
    return normalized_tokens

def stemming_ecs(tokens, stats):
    stemmed_tokens = []
    
    prefixes = ['me', 'ber', 'ter', 'di', 'ke', 'pe', 'se']
    suffixes = ['kan', 'an', 'i', 'nya', 'lah', 'kah']
    
    for token in tokens:
        original_token = token
        stemmed_token = token.lower()
        
        for prefix in sorted(prefixes, key=len, reverse=True):
            if stemmed_token.startswith(prefix) and len(stemmed_token) > len(prefix) + 2:
                stemmed_token = stemmed_token[len(prefix):]
                break
        
        for suffix in sorted(suffixes, key=len, reverse=True):
            if stemmed_token.endswith(suffix) and len(stemmed_token) > len(suffix) + 2:
                stemmed_token = stemmed_token[:-len(suffix)]
                break
        
        if stemmed_token != original_token.lower():
            stats['stemmed_words'] += 1
        
        stemmed_tokens.append(stemmed_token)
    
    return stemmed_tokens