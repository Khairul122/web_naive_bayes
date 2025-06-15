from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify, send_file
from app.models.ScrappingModel import TwitterScraping
from app.extension import db
from datetime import datetime
import subprocess
import pandas as pd
import os
import tempfile
import logging
import numpy as np

scrapping_bp = Blueprint('scrapping', __name__, url_prefix='/scrapping')

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

@scrapping_bp.route('/')
@login_required
def index():
    page = request.args.get('page', 1, type=int)
    tweets = TwitterScraping.query.filter_by(scraped_by=session['user_id'])\
        .order_by(TwitterScraping.scraped_at.desc())\
        .paginate(page=page, per_page=20, error_out=False)
    
    total_tweets = TwitterScraping.query.filter_by(scraped_by=session['user_id']).count()
    
    return render_template('scrapping/index.html', tweets=tweets, total_tweets=total_tweets)

@scrapping_bp.route('/scrape', methods=['POST'])
@login_required
def scrape():
    search_keyword = request.form.get('search_keyword')
    limit = request.form.get('limit', 1000, type=int)
    auth_token = request.form.get('auth_token')
    include_replies = request.form.get('include_replies') == '1'
    include_retweets = request.form.get('include_retweets') == '1'
    
    logger.info(f"Starting scrape process for user {session['user_id']}")
    logger.info(f"Keyword: {search_keyword}, Limit: {limit}")
    
    if not search_keyword or not auth_token:
        logger.warning("Missing search keyword or auth token")
        flash('Keyword dan Auth Token wajib diisi', 'danger')
        return redirect(url_for('scrapping.index'))
    
    try:
        import shutil
        from flask import current_app
        
        dataset_dir = os.path.join(current_app.root_path, 'static', 'dataset')
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            logger.info(f"Created dataset directory: {dataset_dir}")
        
        npx_path = shutil.which('npx')
        if not npx_path:
            logger.error("Node.js/NPX not found in system")
            flash('Node.js tidak terinstall. Silakan install Node.js terlebih dahulu dari https://nodejs.org/', 'danger')
            return redirect(url_for('scrapping.index'))
        
        filename = f"scrape_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        enhanced_query = build_enhanced_search_query(search_keyword, include_replies, include_retweets)
        
        logger.info(f"Using filename: {filename}")
        logger.info(f"Enhanced search query: {enhanced_query}")
        
        cmd = [
            npx_path, '-y', 'tweet-harvest@2.6.1',
            '-o', filename,
            '-s', enhanced_query,
            '--tab', 'LATEST',
            '-l', str(limit),
            '--token', auth_token
        ]
        
        logger.info(f"Executing command in dataset directory: {dataset_dir}")
        logger.info(f"Command: npx -y tweet-harvest@2.6.1 -o {filename} -s '{enhanced_query}' --tab LATEST -l {limit} --token [HIDDEN]")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=dataset_dir, shell=True, timeout=600)
        
        logger.info(f"Command completed with return code: {result.returncode}")
        if result.stdout:
            logger.info(f"Command stdout: {result.stdout}")
        if result.stderr:
            logger.info(f"Command stderr: {result.stderr}")
        
        if result.returncode != 0:
            logger.error(f"Scraping failed: {result.stderr}")
            flash(f'Scraping gagal: {result.stderr or result.stdout}', 'danger')
            return redirect(url_for('scrapping.index'))
        
        csv_path_tweets_data = os.path.join(dataset_dir, 'tweets-data', filename)
        csv_path_direct = os.path.join(dataset_dir, filename)
        
        csv_path = None
        if os.path.exists(csv_path_tweets_data):
            csv_path = csv_path_tweets_data
            logger.info(f"CSV file found at: {csv_path_tweets_data}")
        elif os.path.exists(csv_path_direct):
            csv_path = csv_path_direct
            logger.info(f"CSV file found at: {csv_path_direct}")
        else:
            logger.warning("CSV file not found in expected locations")
            
            files_in_dataset = os.listdir(dataset_dir) if os.path.exists(dataset_dir) else []
            logger.info(f"Files in dataset dir: {files_in_dataset}")
            
            tweets_data_dir = os.path.join(dataset_dir, 'tweets-data')
            if os.path.exists(tweets_data_dir):
                files_in_tweets_data = os.listdir(tweets_data_dir)
                logger.info(f"Files in tweets-data dir: {files_in_tweets_data}")
                
                for file in files_in_tweets_data:
                    if file.endswith('.csv'):
                        csv_path = os.path.join(tweets_data_dir, file)
                        logger.info(f"Found CSV file: {csv_path}")
                        break
            
            if not csv_path:
                flash('Scraping selesai tapi file CSV tidak ditemukan', 'warning')
                return redirect(url_for('scrapping.index'))
        
        if csv_path:
            logger.info(f"Reading CSV file from: {csv_path}")
            df = pd.read_csv(csv_path)
            logger.info(f"Read {len(df)} rows from CSV")
            
            if len(df) > 0:
                df = df.drop_duplicates(subset=['id_str'], keep='first')
                logger.info(f"After removing duplicates: {len(df)} unique tweets")
                
                final_dataset_path = os.path.join(dataset_dir, f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                df.to_csv(final_dataset_path, index=False)
                logger.info(f"Saved final dataset to: {final_dataset_path}")
                
                saved_count = save_tweets_to_db(df, session['user_id'])
                flash(f'Scraping berhasil! Mengumpulkan {saved_count} tweets unik. Dataset tersimpan di static/dataset/', 'success')
            else:
                logger.warning("CSV file is empty")
                flash('File CSV ditemukan tapi kosong', 'warning')
        
    except subprocess.TimeoutExpired:
        logger.error("Scraping timeout (10 minutes)")
        flash('Scraping timeout. Coba kurangi jumlah target atau coba lagi nanti.', 'warning')
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        flash('Node.js atau NPX tidak ditemukan. Pastikan Node.js sudah terinstall.', 'danger')
    except Exception as e:
        logger.error(f"Unexpected error during scraping: {e}")
        flash(f'Error: {str(e)}', 'danger')
    
    return redirect(url_for('scrapping.index'))

@scrapping_bp.route('/export')
@login_required
def export():
    tweets = TwitterScraping.query.filter_by(scraped_by=session['user_id']).all()
    
    if not tweets:
        flash('Tidak ada data untuk di export', 'warning')
        return redirect(url_for('scrapping.index'))
    
    try:
        from flask import current_app
        
        dataset_dir = os.path.join(current_app.root_path, 'static', 'dataset')
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        
        data = []
        for tweet in tweets:
            data.append({
                'tweet_id': tweet.tweet_id_str,
                'username': tweet.username,
                'full_text': tweet.full_text,
                'favorite_count': tweet.favorite_count,
                'retweet_count': tweet.retweet_count,
                'reply_count': tweet.reply_count,
                'created_at': tweet.created_at,
                'tweet_url': tweet.tweet_url,
                'scraped_at': tweet.scraped_at
            })
        
        df = pd.DataFrame(data)
        
        filename = f"export_twitter_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        export_path = os.path.join(dataset_dir, filename)
        
        df.to_csv(export_path, index=False)
        logger.info(f"Data exported to: {export_path}")
        
        return send_file(export_path, as_attachment=True, download_name=filename, mimetype='text/csv')
        
    except Exception as e:
        logger.error(f"Export error: {e}")
        flash(f'Error saat export: {str(e)}', 'danger')
        return redirect(url_for('scrapping.index'))

@scrapping_bp.route('/reset', methods=['POST'])
@login_required
def reset():
    TwitterScraping.query.filter_by(scraped_by=session['user_id']).delete()
    db.session.commit()
    flash('Semua data berhasil dihapus', 'success')
    return redirect(url_for('scrapping.index'))

@scrapping_bp.route('/scrape_demo', methods=['POST'])
@login_required
def scrape_demo():
    search_keyword = request.form.get('search_keyword')
    limit = request.form.get('limit', 100, type=int)
    
    logger.info(f"Starting demo scrape for user {session['user_id']} with limit {limit}")
    
    if not search_keyword:
        flash('Keyword wajib diisi', 'danger')
        return redirect(url_for('scrapping.index'))
    
    try:
        import random
        from datetime import timedelta
        
        demo_tweets = []
        for i in range(min(limit, 20)):
            tweet_data = {
                'id_str': f"177458746136218875{i}",
                'username': f"user_{random.randint(1000, 9999)}",
                'full_text': f"Demo tweet #{i+1} tentang {search_keyword}. Ini adalah contoh tweet untuk testing aplikasi scraping.",
                'favorite_count': random.randint(0, 100),
                'retweet_count': random.randint(0, 50),
                'reply_count': random.randint(0, 20),
                'created_at': (datetime.now() - timedelta(hours=random.randint(1, 72))).strftime('%a %b %d %H:%M:%S +0000 %Y'),
                'tweet_url': f"https://twitter.com/user/status/177458746136218875{i}"
            }
            demo_tweets.append(tweet_data)
        
        logger.info(f"Generated {len(demo_tweets)} demo tweets")
        
        df = pd.DataFrame(demo_tweets)
        saved_count = save_tweets_to_db(df, session['user_id'])
        flash(f'Demo berhasil! Menambahkan {saved_count} tweets demo', 'success')
        
    except Exception as e:
        logger.error(f"Demo scraping error: {e}")
        flash(f'Error demo: {str(e)}', 'danger')
    
    return redirect(url_for('scrapping.index'))

def build_enhanced_search_query(keyword, include_replies=True, include_retweets=True):
    """Build comprehensive search query untuk mendapatkan semua jenis konten"""
    
    base_keyword = keyword.strip()
    
    if 'since:' not in base_keyword and 'until:' not in base_keyword:
        base_keyword += ' since:2024-10-20 until:2025-06-15'
    
    if 'lang:' not in base_keyword:
        base_keyword += ' lang:id'
    
    query_parts = []
    
    base_query = base_keyword
    if not include_retweets:
        base_query += ' -filter:retweets'
    
    query_parts.append(f"({base_query})")
    
    if include_replies:
        reply_query = base_keyword + ' filter:replies'
        if not include_retweets:
            reply_query += ' -filter:retweets'
        query_parts.append(f"({reply_query})")
        logger.info("Including replies in search")
    
    if include_retweets:
        logger.info("Including retweets in search")
    
    media_query = base_keyword + ' filter:media'
    query_parts.append(f"({media_query})")
    logger.info("Including media posts in search")
    
    quote_query = base_keyword + ' filter:quote'
    if not include_retweets:
        quote_query += ' -filter:retweets'
    query_parts.append(f"({quote_query})")
    logger.info("Including quote tweets in search")
    
    combined_query = ' OR '.join(query_parts)
    
    logger.info(f"Built comprehensive search query with {len(query_parts)} parts")
    logger.info(f"Final query: {combined_query}")
    return combined_query

def save_tweets_to_db(df, user_id):
    logger.info(f"Starting to save {len(df)} tweets to database for user {user_id}")
    saved_count = 0
    error_count = 0
    
    for index, row in df.iterrows():
        try:
            logger.debug(f"Processing row {index + 1}/{len(df)}")
            
            tweet_id = str(row.get('id_str', ''))
            if not tweet_id or tweet_id == 'nan':
                logger.warning(f"Row {index + 1}: Invalid tweet_id_str, skipping")
                error_count += 1
                continue
            
            existing_tweet = TwitterScraping.query.filter_by(tweet_id_str=tweet_id).first()
            if existing_tweet:
                logger.debug(f"Row {index + 1}: Tweet {tweet_id} already exists, skipping")
                continue
            
            username = row.get('username')
            if pd.isna(username) or username is None or str(username).lower() == 'nan':
                username = None
            else:
                username = str(username)
            
            user_id_str = row.get('user_id_str')
            if pd.isna(user_id_str) or str(user_id_str).lower() == 'nan':
                user_id_str = None
            else:
                user_id_str = str(user_id_str)
            
            conversation_id_str = row.get('conversation_id_str')
            if pd.isna(conversation_id_str) or str(conversation_id_str).lower() == 'nan':
                conversation_id_str = None
            else:
                conversation_id_str = str(conversation_id_str)
            
            full_text = row.get('full_text', '')
            if pd.isna(full_text):
                logger.warning(f"Row {index + 1}: full_text is NaN, skipping")
                error_count += 1
                continue
            
            favorite_count = row.get('favorite_count', 0)
            if pd.isna(favorite_count):
                favorite_count = 0
            else:
                favorite_count = int(favorite_count)
            
            retweet_count = row.get('retweet_count', 0)
            if pd.isna(retweet_count):
                retweet_count = 0
            else:
                retweet_count = int(retweet_count)
            
            reply_count = row.get('reply_count', 0)
            if pd.isna(reply_count):
                reply_count = 0
            else:
                reply_count = int(reply_count)
            
            quote_count = row.get('quote_count', 0)
            if pd.isna(quote_count):
                quote_count = 0
            else:
                quote_count = int(quote_count)
            
            try:
                created_at_str = row.get('created_at', '')
                if pd.isna(created_at_str):
                    created_at = datetime.utcnow()
                else:
                    created_at = datetime.strptime(str(created_at_str), '%a %b %d %H:%M:%S %z %Y')
                    created_at = created_at.replace(tzinfo=None)
            except:
                created_at = datetime.utcnow()
            
            tweet_url = row.get('tweet_url')
            if pd.isna(tweet_url):
                tweet_url = None
            else:
                tweet_url = str(tweet_url)
            
            in_reply_to_screen_name = row.get('in_reply_to_screen_name')
            if pd.isna(in_reply_to_screen_name) or str(in_reply_to_screen_name).lower() == 'nan':
                in_reply_to_screen_name = None
            else:
                in_reply_to_screen_name = str(in_reply_to_screen_name)
            
            in_reply_to_tweet_id = row.get('in_reply_to_status_id_str')
            if pd.isna(in_reply_to_tweet_id) or str(in_reply_to_tweet_id).lower() == 'nan':
                in_reply_to_tweet_id = None
            else:
                in_reply_to_tweet_id = str(in_reply_to_tweet_id)
            
            is_retweet = str(full_text).startswith('RT @')
            is_reply = in_reply_to_screen_name is not None
            is_quote = row.get('quoted_status_id_str') is not None and not pd.isna(row.get('quoted_status_id_str'))
            
            image_url = row.get('image_url')
            has_media = False
            media_urls = None
            media_type = None
            
            if not pd.isna(image_url) and str(image_url) != 'nan':
                has_media = True
                media_urls = str(image_url)
                if 'video' in str(image_url).lower():
                    media_type = 'video'
                elif any(ext in str(image_url).lower() for ext in ['.jpg', '.png', '.gif', '.jpeg']):
                    media_type = 'image'
                else:
                    media_type = 'media'
            
            lang = row.get('lang', 'id')
            if pd.isna(lang):
                lang = 'id'
            
            tweet = TwitterScraping(
                tweet_id_str=tweet_id,
                conversation_id_str=conversation_id_str,
                username=username,
                user_id_str=user_id_str,
                full_text=str(full_text),
                favorite_count=favorite_count,
                retweet_count=retweet_count,
                reply_count=reply_count,
                quote_count=quote_count,
                created_at=created_at,
                tweet_url=tweet_url,
                is_retweet=is_retweet,
                is_reply=is_reply,
                is_quote=is_quote,
                in_reply_to_screen_name=in_reply_to_screen_name,
                in_reply_to_tweet_id=in_reply_to_tweet_id,
                has_media=has_media,
                media_urls=media_urls,
                media_type=media_type,
                lang=str(lang),
                scraped_by=user_id
            )
            
            db.session.add(tweet)
            saved_count += 1
            
            if (index + 1) % 10 == 0:
                logger.info(f"Processed {index + 1}/{len(df)} rows, {saved_count} prepared for saving")
                
        except Exception as e:
            logger.error(f"Row {index + 1}: Error processing tweet: {e}")
            error_count += 1
            continue
    
    try:
        db.session.commit()
        logger.info(f"Successfully saved {saved_count} tweets to database")
        
        retweet_count = db.session.query(TwitterScraping).filter_by(scraped_by=user_id, is_retweet=True).count()
        reply_count = db.session.query(TwitterScraping).filter_by(scraped_by=user_id, is_reply=True).count()
        quote_count = db.session.query(TwitterScraping).filter_by(scraped_by=user_id, is_quote=True).count()
        media_count = db.session.query(TwitterScraping).filter_by(scraped_by=user_id, has_media=True).count()
        
        logger.info(f"Content breakdown - Retweets: {retweet_count}, Replies: {reply_count}, Quotes: {quote_count}, Media: {media_count}")
        
        if error_count > 0:
            logger.warning(f"Encountered {error_count} errors during processing")
    except Exception as e:
        logger.error(f"Database commit failed: {e}")
        db.session.rollback()
        raise e
    
    return saved_count