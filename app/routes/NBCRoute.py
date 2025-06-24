from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from app.models.NBCModel import NBCTraining, NBCTesting, NBCModel as NBCModelData
from app.models.KonversiModel import TfidfConversion
from app.models.SentimenModel import SentimentAnalysis
from app.extension import db
from datetime import datetime
import pandas as pd
import numpy as np
import json
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import io
import base64
from collections import Counter
import math

nbc_bp = Blueprint('nbc', __name__, url_prefix='/nbc')

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

def create_confusion_matrix_plot(y_true, y_pred, classes):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    plot_data = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return plot_data

def create_performance_metrics_plot(metrics_data):
    plt.figure(figsize=(10, 6))
    
    classes = list(metrics_data.keys())
    precision_scores = [metrics_data[cls]['precision'] for cls in classes]
    recall_scores = [metrics_data[cls]['recall'] for cls in classes]
    f1_scores = [metrics_data[cls]['f1-score'] for cls in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    plt.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
    plt.bar(x, recall_scores, width, label='Recall', alpha=0.8)
    plt.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Performance Metrics by Class')
    plt.xticks(x, classes)
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    plot_data = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return plot_data

def create_wordcloud_plot(text_data, sentiment_label):
    if not text_data:
        return None
    
    # Filter teks yang valid dan gabungkan
    valid_texts = [str(text).strip() for text in text_data if text and str(text).strip()]
    if not valid_texts:
        return None
        
    combined_text = ' '.join(valid_texts)
    
    # Pastikan ada teks yang cukup
    if len(combined_text.strip()) < 10:
        return None
    
    try:
        if sentiment_label == 'positif':
            colormap = 'Greens'
        elif sentiment_label == 'negatif':
            colormap = 'Reds'
        else:
            colormap = 'Blues'
        
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap=colormap,
            max_words=100,
            min_font_size=10,
            prefer_horizontal=0.7
        ).generate(combined_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud - {sentiment_label.capitalize()}')
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plot_data = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return plot_data
    except Exception as e:
        logger.error(f"Error generating wordcloud for {sentiment_label}: {e}")
        return None

def calculate_manual_naive_bayes(X_train, y_train, X_test, classes, alpha=1.0):
    manual_calculations = []
    
    n_samples = len(y_train)
    class_counts = Counter(y_train)
    
    class_probabilities = {}
    for cls in classes:
        class_probabilities[cls] = (class_counts[cls] + alpha) / (n_samples + alpha * len(classes))
    
    feature_probabilities = {}
    for cls in classes:
        class_indices = [i for i, label in enumerate(y_train) if label == cls]
        class_features = X_train[class_indices]
        
        feature_counts = np.sum(class_features, axis=0)
        total_features = np.sum(feature_counts)
        
        feature_probabilities[cls] = (feature_counts + alpha) / (total_features + alpha * len(feature_counts))
    
    for test_idx, test_sample in enumerate(X_test[:5]):
        calculation = {
            'test_index': test_idx,
            'features': test_sample.tolist(),
            'class_calculations': {}
        }
        
        for cls in classes:
            log_prob = math.log(class_probabilities[cls])
            
            feature_log_probs = []
            for feature_idx, feature_value in enumerate(test_sample):
                if feature_value > 0:
                    feature_prob = feature_probabilities[cls][feature_idx]
                    feature_log_prob = feature_value * math.log(feature_prob)
                    log_prob += feature_log_prob
                    feature_log_probs.append({
                        'feature_index': feature_idx,
                        'feature_value': feature_value,
                        'feature_probability': feature_prob,
                        'log_contribution': feature_log_prob
                    })
            
            calculation['class_calculations'][cls] = {
                'prior_probability': class_probabilities[cls],
                'log_prior': math.log(class_probabilities[cls]),
                'feature_contributions': feature_log_probs,
                'total_log_probability': log_prob
            }
        
        class_scores = {cls: calc['total_log_probability'] 
                       for cls, calc in calculation['class_calculations'].items()}
        predicted_class = max(class_scores, key=class_scores.get)
        calculation['predicted_class'] = predicted_class
        calculation['class_scores'] = class_scores
        
        manual_calculations.append(calculation)
    
    return manual_calculations

@nbc_bp.route('/')
@login_required
def index():
    user_id = session['user_id']
    
    total_tfidf = TfidfConversion.query.filter_by(converted_by=user_id).count()
    training_count = NBCTraining.query.filter_by(user_id=user_id).count()
    testing_count = NBCTesting.query.filter_by(user_id=user_id).count()
    model_count = NBCModelData.query.filter_by(user_id=user_id).count()
    
    latest_model = NBCModelData.query.filter_by(user_id=user_id)\
        .order_by(NBCModelData.created_at.desc()).first()
    
    accuracy = latest_model.accuracy if latest_model else 0
    
    page = request.args.get('page', 1, type=int)
    training_data = NBCTraining.query.filter_by(user_id=user_id)\
        .order_by(NBCTraining.created_at.desc())\
        .paginate(page=page, per_page=20, error_out=False)
    
    stats = {
        'total_tfidf': total_tfidf,
        'training_count': training_count,
        'testing_count': testing_count,
        'model_count': model_count,
        'accuracy': round(accuracy, 2) if accuracy else 0,
        'training_rate': round((training_count / total_tfidf * 100), 2) if total_tfidf > 0 else 0
    }
    
    return render_template('nbc/index.html', training_data=training_data, stats=stats, latest_model=latest_model)

@nbc_bp.route('/split', methods=['POST'])
@login_required
def split_data():
    user_id = session['user_id']
    
    test_size = float(request.form.get('test_size', 0.3))
    random_state = int(request.form.get('random_state', 42))
    
    logger.info(f"Memulai split data untuk user {user_id}")
    logger.info(f"Parameter: test_size={test_size}, random_state={random_state}")
    
    tfidf_data = db.session.query(TfidfConversion, SentimentAnalysis)\
        .join(SentimentAnalysis, TfidfConversion.sentiment_id == SentimentAnalysis.id)\
        .filter(TfidfConversion.converted_by == user_id)\
        .all()
    
    logger.info(f"Ditemukan {len(tfidf_data)} data TF-IDF")
    
    if not tfidf_data:
        logger.warning("Tidak ada data TF-IDF untuk digunakan")
        flash('Tidak ada data TF-IDF untuk digunakan', 'warning')
        return redirect(url_for('nbc.index'))
    
    if len(tfidf_data) < 10:
        logger.warning(f"Data tidak cukup: {len(tfidf_data)} (minimal 10)")
        flash('Minimal 10 data diperlukan untuk split training/testing', 'warning')
        return redirect(url_for('nbc.index'))
    
    try:
        logger.info("Memproses data untuk split...")
        X = []
        y = []
        conversion_ids = []
        
        for conversion, sentiment in tfidf_data:
            feature_vector = json.loads(conversion.feature_vector)
            X.append(feature_vector)
            y.append(sentiment.sentiment_label)
            conversion_ids.append(conversion.id)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Shape data: X={X.shape}, y={len(y)}")
        
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y, conversion_ids, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Hasil split: training={len(ids_train)}, testing={len(ids_test)}")
        
        logger.info("Menghapus data split sebelumnya...")
        NBCTraining.query.filter_by(user_id=user_id).delete()
        NBCTesting.query.filter_by(user_id=user_id).delete()
        
        logger.info("Menyimpan data training...")
        for i, conversion_id in enumerate(ids_train):
            training_record = NBCTraining(
                user_id=user_id,
                conversion_id=conversion_id,
                feature_vector=json.dumps(X_train[i].tolist()),
                label=y_train[i],
                split_method='train_test_split',
                test_size=test_size,
                random_state=random_state
            )
            db.session.add(training_record)
        
        logger.info("Menyimpan data testing...")
        for i, conversion_id in enumerate(ids_test):
            testing_record = NBCTesting(
                user_id=user_id,
                conversion_id=conversion_id,
                feature_vector=json.dumps(X_test[i].tolist()),
                true_label=y_test[i],
                split_method='train_test_split',
                test_size=test_size,
                random_state=random_state
            )
            db.session.add(testing_record)
        
        db.session.commit()
        logger.info("Split data berhasil disimpan ke database")
        flash(f'Data berhasil di-split: {len(ids_train)} training, {len(ids_test)} testing', 'success')
        
    except Exception as e:
        logger.error(f"Error split data: {e}")
        flash(f'Error split data: {str(e)}', 'danger')
    
    return redirect(url_for('nbc.index'))

@nbc_bp.route('/train', methods=['POST'])
@login_required
def train_model():
    user_id = session['user_id']
    
    alpha = float(request.form.get('alpha', 1.0))
    
    logger.info(f"Memulai training model untuk user {user_id}")
    logger.info(f"Parameter alpha: {alpha}")
    
    training_data = NBCTraining.query.filter_by(user_id=user_id).all()
    
    logger.info(f"Ditemukan {len(training_data)} data training")
    
    if not training_data:
        logger.warning("Tidak ada data training")
        flash('Tidak ada data training. Lakukan split data terlebih dahulu', 'warning')
        return redirect(url_for('nbc.index'))
    
    if len(training_data) < 5:
        logger.warning(f"Data training tidak cukup: {len(training_data)} (minimal 5)")
        flash('Minimal 5 data training diperlukan', 'warning')
        return redirect(url_for('nbc.index'))
    
    try:
        logger.info("Memproses data training...")
        X_train = []
        y_train = []
        
        for data in training_data:
            feature_vector = json.loads(data.feature_vector)
            X_train.append(feature_vector)
            y_train.append(data.label)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        logger.info(f"Shape data training: X={X_train.shape}, y={len(y_train)}")
        
        logger.info("Melatih model Multinomial Naive Bayes...")
        model = MultinomialNB(alpha=alpha)
        model.fit(X_train, y_train)
        
        logger.info("Model berhasil dilatih, menyimpan parameter...")
        feature_log_prob = model.feature_log_prob_.tolist()
        class_log_prior = model.class_log_prior_.tolist()
        classes = model.classes_.tolist()
        
        logger.info(f"Jumlah fitur: {X_train.shape[1]}")
        logger.info(f"Jumlah kelas: {len(classes)}")
        logger.info(f"Kelas: {classes}")
        
        logger.info("Menghapus model sebelumnya...")
        NBCModelData.query.filter_by(user_id=user_id).delete()
        
        model_name = f'NBC_Model_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        logger.info(f"Menyimpan model: {model_name}")
        
        model_record = NBCModelData(
            user_id=user_id,
            model_name=model_name,
            alpha=alpha,
            feature_log_prob=json.dumps(feature_log_prob),
            class_log_prior=json.dumps(class_log_prior),
            classes=json.dumps(classes),
            n_features=X_train.shape[1],
            n_classes=len(classes),
            training_samples=len(training_data),
            training_method='MultinomialNB'
        )
        
        db.session.add(model_record)
        db.session.commit()
        
        logger.info("Model berhasil disimpan ke database")
        flash(f'Model NBC berhasil dilatih dengan {len(training_data)} data training', 'success')
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        flash(f'Error training model: {str(e)}', 'danger')
    
    return redirect(url_for('nbc.index'))

@nbc_bp.route('/test', methods=['POST'])
@login_required
def test_model():
    user_id = session['user_id']
    
    logger.info(f"Memulai testing model untuk user {user_id}")
    
    model_data = NBCModelData.query.filter_by(user_id=user_id).first()
    if not model_data:
        logger.warning("Model belum tersedia")
        flash('Model belum dilatih. Lakukan training terlebih dahulu', 'warning')
        return redirect(url_for('nbc.index'))
    
    logger.info(f"Menggunakan model: {model_data.model_name}")
    
    testing_data = NBCTesting.query.filter_by(user_id=user_id).all()
    if not testing_data:
        logger.warning("Tidak ada data testing")
        flash('Tidak ada data testing. Lakukan split data terlebih dahulu', 'warning')
        return redirect(url_for('nbc.index'))
    
    logger.info(f"Ditemukan {len(testing_data)} data testing")
    
    try:
        logger.info("Memproses data testing...")
        X_test = []
        y_test = []
        
        for data in testing_data:
            feature_vector = json.loads(data.feature_vector)
            X_test.append(feature_vector)
            y_test.append(data.true_label)
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        logger.info(f"Shape data testing: X={X_test.shape}, y={len(y_test)}")
        
        logger.info("Membangun ulang model dari parameter tersimpan...")
        model = MultinomialNB(alpha=model_data.alpha)
        model.feature_log_prob_ = np.array(json.loads(model_data.feature_log_prob))
        model.class_log_prior_ = np.array(json.loads(model_data.class_log_prior))
        model.classes_ = np.array(json.loads(model_data.classes))
        
        logger.info("Melakukan prediksi...")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        logger.info("Menghitung akurasi...")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        logger.info(f"Akurasi model: {accuracy:.4f} ({accuracy:.2%})")
        
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        
        logger.info("Menyimpan hasil prediksi...")
        correct_count = 0
        for i, data in enumerate(testing_data):
            data.predicted_label = y_pred[i]
            data.prediction_probability = json.dumps(y_prob[i].tolist())
            is_correct = (y_test[i] == y_pred[i])
            data.is_correct = is_correct
            if is_correct:
                correct_count += 1
        
        logger.info(f"Prediksi benar: {correct_count}/{len(testing_data)}")
        
        model_data.accuracy = accuracy
        model_data.precision_score = precision
        model_data.recall_score = recall
        model_data.f1_score = f1
        model_data.classification_report = json.dumps(classification_rep)
        model_data.testing_samples = len(testing_data)
        model_data.tested_at = datetime.utcnow()
        
        db.session.commit()
        logger.info("Hasil testing berhasil disimpan ke database")
        
        flash(f'Model berhasil diuji dengan akurasi {accuracy:.2%}', 'success')
        
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        flash(f'Error testing model: {str(e)}', 'danger')
    
    return redirect(url_for('nbc.index'))

@nbc_bp.route('/results')
@login_required
def results():
    user_id = session['user_id']
    
    model_data = NBCModelData.query.filter_by(user_id=user_id).first()
    if not model_data:
        flash('Model belum tersedia', 'warning')
        return redirect(url_for('nbc.index'))
    
    page = request.args.get('page', 1, type=int)
    testing_results = NBCTesting.query.filter_by(user_id=user_id)\
        .filter(NBCTesting.predicted_label.isnot(None))\
        .order_by(NBCTesting.created_at.desc())\
        .paginate(page=page, per_page=20, error_out=False)
    
    evaluation_data = None
    wordclouds = None
    manual_calculations = None
    

    has_classification_report = hasattr(model_data, 'classification_report') and model_data.classification_report
    
    if testing_results.items and has_classification_report:
        try:
            testing_data = NBCTesting.query.filter_by(user_id=user_id)\
                .filter(NBCTesting.predicted_label.isnot(None)).all()
            
            y_true = [data.true_label for data in testing_data]
            y_pred = [data.predicted_label for data in testing_data]
            classes = json.loads(model_data.classes)
            
            confusion_matrix_plot = create_confusion_matrix_plot(y_true, y_pred, classes)
            
            classification_rep = json.loads(model_data.classification_report)
            class_metrics = {cls: classification_rep[cls] for cls in classes if cls in classification_rep}
            performance_plot = create_performance_metrics_plot(class_metrics)
            
            evaluation_data = {
                'confusion_matrix': confusion_matrix_plot,
                'performance_metrics': performance_plot,
                'classification_report': class_metrics
            }
        except Exception as e:
            logger.error(f"Error creating evaluation plots: {e}")
    elif testing_results.items:
        try:
            testing_data = NBCTesting.query.filter_by(user_id=user_id)\
                .filter(NBCTesting.predicted_label.isnot(None)).all()
            
            y_true = [data.true_label for data in testing_data]
            y_pred = [data.predicted_label for data in testing_data]
            classes = json.loads(model_data.classes)
            
            confusion_matrix_plot = create_confusion_matrix_plot(y_true, y_pred, classes)
            
            from sklearn.metrics import classification_report
            classification_rep = classification_report(y_true, y_pred, output_dict=True)
            class_metrics = {cls: classification_rep[cls] for cls in classes if cls in classification_rep}
            performance_plot = create_performance_metrics_plot(class_metrics)
            
            evaluation_data = {
                'confusion_matrix': confusion_matrix_plot,
                'performance_metrics': performance_plot,
                'classification_report': class_metrics
            }
        except Exception as e:
            logger.error(f"Error creating evaluation plots: {e}")
    
    try:

        sentiment_data = db.session.query(SentimentAnalysis)\
            .filter(SentimentAnalysis.labeled_by == user_id)\
            .all()
        
        if sentiment_data:
            wordclouds = {}
            sentiment_groups = {}
            
            for data in sentiment_data:
                label = data.sentiment_label
                if label not in sentiment_groups:
                    sentiment_groups[label] = []
                text_to_use = data.processed_text or data.tweet_text or ''
                if text_to_use and text_to_use.strip():
                    sentiment_groups[label].append(text_to_use)
            
            for label, texts in sentiment_groups.items():
                if texts: 
                    wordcloud_plot = create_wordcloud_plot(texts, label)
                    if wordcloud_plot:
                        wordclouds[label] = wordcloud_plot
        else:
            sentiment_data_fallback = db.session.query(SentimentAnalysis)\
                .join(TfidfConversion, SentimentAnalysis.id == TfidfConversion.sentiment_id)\
                .filter(TfidfConversion.converted_by == user_id)\
                .all()
            
            if sentiment_data_fallback:
                wordclouds = {}
                sentiment_groups = {}
                
                for data in sentiment_data_fallback:
                    label = data.sentiment_label
                    if label not in sentiment_groups:
                        sentiment_groups[label] = []
                    text_to_use = data.processed_text or data.tweet_text or ''
                    if text_to_use and text_to_use.strip():
                        sentiment_groups[label].append(text_to_use)
                
                for label, texts in sentiment_groups.items():
                    if texts:
                        wordcloud_plot = create_wordcloud_plot(texts, label)
                        if wordcloud_plot:
                            wordclouds[label] = wordcloud_plot
    except Exception as e:
        logger.error(f"Error creating wordclouds: {e}")
        logger.info("Debug: Mencoba mencari data sentimen dengan query alternatif")
    
    try:
        training_data = NBCTraining.query.filter_by(user_id=user_id).all()
        testing_data = NBCTesting.query.filter_by(user_id=user_id).all()
        
        if training_data and testing_data:
            X_train = []
            y_train = []
            for data in training_data:
                feature_vector = json.loads(data.feature_vector)
                X_train.append(feature_vector)
                y_train.append(data.label)
            
            X_test = []
            for data in testing_data[:5]:
                feature_vector = json.loads(data.feature_vector)
                X_test.append(feature_vector)
            
            X_train = np.array(X_train)
            X_test = np.array(X_test)
            classes = json.loads(model_data.classes)
            
            manual_calculations = calculate_manual_naive_bayes(
                X_train, y_train, X_test, classes, model_data.alpha
            )
    except Exception as e:
        logger.error(f"Error creating manual calculations: {e}")
    
    return render_template('nbc/results.html',
                         testing_results=testing_results,
                         model_data=model_data,
                         evaluation_data=evaluation_data,
                         wordclouds=wordclouds,
                         manual_calculations=manual_calculations)

@nbc_bp.route('/evaluation')
@login_required
def evaluation():
    user_id = session['user_id']
    
    model_data = NBCModelData.query.filter_by(user_id=user_id).first()
    if not model_data or not model_data.classification_report:
        flash('Lakukan testing model terlebih dahulu', 'warning')
        return redirect(url_for('nbc.index'))
    
    testing_data = NBCTesting.query.filter_by(user_id=user_id)\
        .filter(NBCTesting.predicted_label.isnot(None)).all()
    
    if not testing_data:
        flash('Tidak ada hasil testing untuk dievaluasi', 'warning')
        return redirect(url_for('nbc.index'))
    
    y_true = [data.true_label for data in testing_data]
    y_pred = [data.predicted_label for data in testing_data]
    classes = json.loads(model_data.classes)
    
    confusion_matrix_plot = create_confusion_matrix_plot(y_true, y_pred, classes)
    
    classification_rep = json.loads(model_data.classification_report)
    class_metrics = {cls: classification_rep[cls] for cls in classes if cls in classification_rep}
    performance_plot = create_performance_metrics_plot(class_metrics)
    
    evaluation_data = {
        'confusion_matrix': confusion_matrix_plot,
        'performance_metrics': performance_plot,
        'classification_report': class_metrics,
        'overall_metrics': {
            'accuracy': model_data.accuracy,
            'precision': model_data.precision_score,
            'recall': model_data.recall_score,
            'f1_score': model_data.f1_score
        }
    }
    
    return render_template('nbc/evaluation.html', 
                         evaluation_data=evaluation_data,
                         model_data=model_data)

@nbc_bp.route('/wordcloud')
@login_required
def wordcloud():
    user_id = session['user_id']
    
    sentiment_data = db.session.query(SentimentAnalysis)\
        .filter(SentimentAnalysis.analyzed_by == user_id)\
        .all()
    
    if not sentiment_data:
        flash('Tidak ada data sentimen untuk membuat word cloud', 'warning')
        return redirect(url_for('nbc.index'))
    
    wordclouds = {}
    sentiment_groups = {}
    
    for data in sentiment_data:
        label = data.sentiment_label
        if label not in sentiment_groups:
            sentiment_groups[label] = []
        sentiment_groups[label].append(data.cleaned_text)
    
    for label, texts in sentiment_groups.items():
        wordcloud_plot = create_wordcloud_plot(texts, label)
        if wordcloud_plot:
            wordclouds[label] = wordcloud_plot
    
    return render_template('nbc/wordcloud.html', 
                         wordclouds=wordclouds,
                         sentiment_groups=sentiment_groups)

@nbc_bp.route('/manual-calculation')
@login_required
def manual_calculation():
    user_id = session['user_id']
    
    model_data = NBCModelData.query.filter_by(user_id=user_id).first()
    if not model_data:
        flash('Model belum tersedia', 'warning')
        return redirect(url_for('nbc.index'))
    
    training_data = NBCTraining.query.filter_by(user_id=user_id).all()
    testing_data = NBCTesting.query.filter_by(user_id=user_id).all()
    
    if not training_data or not testing_data:
        flash('Data training dan testing diperlukan', 'warning')
        return redirect(url_for('nbc.index'))
    
    X_train = []
    y_train = []
    for data in training_data:
        feature_vector = json.loads(data.feature_vector)
        X_train.append(feature_vector)
        y_train.append(data.label)
    
    X_test = []
    for data in testing_data[:5]:
        feature_vector = json.loads(data.feature_vector)
        X_test.append(feature_vector)
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    classes = json.loads(model_data.classes)
    
    manual_calculations = calculate_manual_naive_bayes(
        X_train, y_train, X_test, classes, model_data.alpha
    )
    
    return render_template('nbc/manual_calculation.html',
                         manual_calculations=manual_calculations,
                         model_data=model_data,
                         classes=classes)

@nbc_bp.route('/debug-data')
@login_required
def debug_data():
    user_id = session['user_id']
    sentiment_count = SentimentAnalysis.query.count()
    user_sentiment = SentimentAnalysis.query.filter_by(labeled_by=user_id).count()
    
    tfidf_count = TfidfConversion.query.filter_by(converted_by=user_id).count()

    joined_data = db.session.query(SentimentAnalysis, TfidfConversion)\
        .join(TfidfConversion, SentimentAnalysis.id == TfidfConversion.sentiment_id)\
        .filter(TfidfConversion.converted_by == user_id)\
        .count()
    
    sample_sentiment = SentimentAnalysis.query.filter_by(labeled_by=user_id).first()
    
    debug_info = {
        'total_sentiment_data': sentiment_count,
        'user_sentiment_data_by_labeled_by': user_sentiment,
        'user_tfidf_data': tfidf_count,
        'joined_sentiment_tfidf': joined_data,
        'user_id': user_id,
        'sample_sentiment': {
            'id': sample_sentiment.id if sample_sentiment else None,
            'sentiment_label': sample_sentiment.sentiment_label if sample_sentiment else None,
            'has_processed_text': bool(sample_sentiment.processed_text) if sample_sentiment else False,
            'has_tweet_text': bool(sample_sentiment.tweet_text) if sample_sentiment else False
        } if sample_sentiment else None
    }
    
    return f"<pre>{debug_info}</pre>"

@nbc_bp.route('/reset', methods=['POST'])
@login_required
def reset():
    user_id = session['user_id']
    
    logger.info(f"Memulai reset semua data NBC untuk user {user_id}")
    
    training_count = NBCTraining.query.filter_by(user_id=user_id).count()
    testing_count = NBCTesting.query.filter_by(user_id=user_id).count()
    model_count = NBCModelData.query.filter_by(user_id=user_id).count()
    
    logger.info(f"Data yang akan dihapus: training={training_count}, testing={testing_count}, model={model_count}")
    
    NBCTraining.query.filter_by(user_id=user_id).delete()
    NBCTesting.query.filter_by(user_id=user_id).delete()
    NBCModelData.query.filter_by(user_id=user_id).delete()
    
    db.session.commit()
    logger.info("Semua data NBC berhasil dihapus")
    
    flash('Semua data NBC berhasil dihapus', 'success')
    return redirect(url_for('nbc.index'))