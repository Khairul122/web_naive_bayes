from flask import Blueprint, render_template, session, redirect, url_for
from app.models.ScrappingModel import TwitterScraping
from app.models.SentimenModel import SentimentAnalysis
from app.models.KonversiModel import TfidfConversion
from app.models.NBCModel import NBCModel
from app.extension import db

dashboard_bp = Blueprint('dashboard', __name__, url_prefix='/')

@dashboard_bp.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    user_id = session.get('user_id')
    username = session.get('username', 'User')
    
    try:
        total_tweets = TwitterScraping.query.filter_by(scraped_by=user_id).count()
        sentiment_count = SentimentAnalysis.query.filter_by(labeled_by=user_id).count()
        tfidf_count = TfidfConversion.query.filter_by(converted_by=user_id).count()
        
        latest_model = NBCModel.query.filter_by(user_id=user_id)\
            .order_by(NBCModel.created_at.desc()).first()
        nbc_accuracy = round(latest_model.accuracy * 100, 1) if latest_model and latest_model.accuracy else 0
        
        positif_count = SentimentAnalysis.query.filter_by(labeled_by=user_id, sentiment_label='positif').count()
        negatif_count = SentimentAnalysis.query.filter_by(labeled_by=user_id, sentiment_label='negatif').count()
        netral_count = SentimentAnalysis.query.filter_by(labeled_by=user_id, sentiment_label='netral').count()
        
        stats = {
            'total_tweets': total_tweets,
            'sentiment_count': sentiment_count,
            'tfidf_count': tfidf_count,
            'nbc_accuracy': nbc_accuracy,
            'positif_count': positif_count,
            'negatif_count': negatif_count,
            'netral_count': netral_count,
            'model_trained': latest_model is not None,
            'model_tested': latest_model.tested_at is not None if latest_model else False
        }
        
        recent_activity = []
        
        latest_tweet = TwitterScraping.query.filter_by(scraped_by=user_id)\
            .order_by(TwitterScraping.scraped_at.desc()).first()
        if latest_tweet:
            recent_activity.append({
                'title': 'Tweet Scraping',
                'description': f'Scraping tweet dari @{latest_tweet.username}',
                'time': latest_tweet.scraped_at.strftime('%d/%m/%Y %H:%M'),
                'icon': 'twitter',
                'color': 'primary'
            })
        
        latest_sentiment = SentimentAnalysis.query.filter_by(labeled_by=user_id)\
            .order_by(SentimentAnalysis.created_at.desc()).first()
        if latest_sentiment:
            recent_activity.append({
                'title': 'Sentiment Analysis',
                'description': f'Label {latest_sentiment.sentiment_label} ditambahkan',
                'time': latest_sentiment.created_at.strftime('%d/%m/%Y %H:%M'),
                'icon': 'heart',
                'color': 'warning'
            })

        latest_tfidf = TfidfConversion.query.filter_by(converted_by=user_id)\
            .order_by(TfidfConversion.created_at.desc()).first()
        if latest_tfidf:
            recent_activity.append({
                'title': 'TF-IDF Conversion',
                'description': f'{latest_tfidf.total_features} fitur dikonversi',
                'time': latest_tfidf.created_at.strftime('%d/%m/%Y %H:%M'),
                'icon': 'exchange-alt',
                'color': 'success'
            })
        
        if latest_model:
            status = 'diuji' if latest_model.tested_at else 'dilatih'
            time_ref = latest_model.tested_at or latest_model.trained_at
            recent_activity.append({
                'title': 'NBC Model',
                'description': f'Model {status} dengan akurasi {nbc_accuracy}%',
                'time': time_ref.strftime('%d/%m/%Y %H:%M'),
                'icon': 'brain',
                'color': 'danger'
            })
        
        recent_activity = sorted(recent_activity, 
                               key=lambda x: x['time'], 
                               reverse=True)[:5]
        
    except Exception as e:
        stats = {
            'total_tweets': 0,
            'sentiment_count': 0,
            'tfidf_count': 0,
            'nbc_accuracy': 0,
            'positif_count': 0,
            'negatif_count': 0,
            'netral_count': 0,
            'model_trained': False,
            'model_tested': False
        }
        recent_activity = []
    
    return render_template('dashboard/index.html', 
                         username=username, 
                         stats=stats, 
                         recent_activity=recent_activity)