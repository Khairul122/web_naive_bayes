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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging

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
        logger.info(f"Akurasi model: {accuracy:.4f} ({accuracy:.2%})")
        
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
    
    return render_template('nbc/results.html', 
                         testing_results=testing_results, 
                         model_data=model_data)

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