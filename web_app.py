"""
Face Authentication Web Frontend
Flask server that wraps the existing face auth backend modules.
Provides:
  - MJPEG video stream with face detection overlays
  - User management API (list/add/remove/rename)
  - Face authentication API
  - Two pages: Registration and Home (locked/unlocked)
"""

import os
import sys
import time
import json
import queue
import logging
import threading
import cv2
import numpy as np
import torch
import yaml
from datetime import timedelta
from flask import Flask, render_template, jsonify, request, Response, session, redirect, url_for, stream_with_context

# Phase 4: JWT support (optional — graceful fallback if not installed)
try:
    from flask_jwt_extended import (
        JWTManager, create_access_token, create_refresh_token,
        jwt_required, get_jwt_identity, get_jwt,
    )
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    print("[Web] WARNING: flask-jwt-extended not installed. "
          "JWT session security disabled. "
          "Install with: pip install flask-jwt-extended")
    # Provide no-op decorators so routes don't break
    def jwt_required(optional=False, refresh=False):
        def decorator(fn):
            return fn
        return decorator
    def get_jwt_identity():
        return None
    def get_jwt():
        return {}
    def create_access_token(identity=None, expires_delta=None):
        return ""
    def create_refresh_token(identity=None):
        return ""

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.models import get_model
from src.preprocessing import FaceDetector, enhance_face_illumination
from src.datasets.augmentations import get_val_transforms
from src.auth import (
    FaceEncoder,
    IdentityStore,
    AttemptLogger,
    RegistrationManager,
    AuthenticationManager,
    RiskDecisionEngine,
    IPRateLimiter,
    PINStore,
)
from src.liveness import BlinkChallenge
from src.security import IntegrityChecker

logger = logging.getLogger(__name__)

# Load full config early (for admin credentials + Flask secret)
def _load_full_config():
    cfg_path = os.path.join(os.path.dirname(__file__), 'config', 'config_auth.yaml')
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f) or {}

_full_cfg = _load_full_config()
_admin_cfg = _full_cfg.get('admin', {})
ADMIN_USERNAME = _admin_cfg.get('username', 'admin')
ADMIN_PASSWORD = _admin_cfg.get('password', 'faceauth@2025')

app = Flask(__name__)
# Flask session secret — used for admin login sessions
import secrets
app.secret_key = secrets.token_hex(32)


# =============================================================================
# Phase 4: JWT Configuration
# =============================================================================
# Security: JWT secret from environment variable with auto-generated fallback
_jwt_secret = os.environ.get('FACEAUTH_JWT_SECRET')
if not _jwt_secret:
    import secrets
    _jwt_secret = secrets.token_hex(32)
    print("[Web] WARNING: FACEAUTH_JWT_SECRET not set. Using auto-generated secret. "
          "Set this env var for production use.")

app.config['JWT_SECRET_KEY'] = _jwt_secret
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(minutes=15)
app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(hours=1)
app.config['JWT_TOKEN_LOCATION'] = ['headers']
app.config['JWT_HEADER_NAME'] = 'Authorization'
app.config['JWT_HEADER_TYPE'] = 'Bearer'

# Token blacklist for logout
_jwt_blacklist = set()

if JWT_AVAILABLE:
    jwt = JWTManager(app)

    @jwt.token_in_blocklist_loader
    def check_token_blacklist(jwt_header, jwt_payload):
        """Check if token has been revoked (logged out)."""
        jti = jwt_payload.get('jti', '')
        return jti in _jwt_blacklist

    @jwt.revoked_token_loader
    def revoked_token_response(jwt_header, jwt_payload):
        return jsonify({
            'status': 'token_revoked',
            'message': 'Token has been revoked. Please authenticate again.'
        }), 401

    @jwt.expired_token_loader
    def expired_token_response(jwt_header, jwt_payload):
        logger.info(f"Expired token used by: {jwt_payload.get('sub', 'unknown')}")
        return jsonify({
            'status': 'token_expired',
            'message': 'Token has expired. Please authenticate again.'
        }), 401

    @jwt.invalid_token_loader
    def invalid_token_response(error_string):
        logger.warning(f"Invalid token access attempt: {error_string}")
        return jsonify({
            'status': 'invalid_token',
            'message': 'Invalid token.'
        }), 401


# =============================================================================
# Global state
# =============================================================================
class AppState:
    """Thread-safe application state."""

    def __init__(self):
        self.lock = threading.Lock()
        self.camera = None
        self.camera_lock = threading.Lock()

        # Auth state
        self.is_unlocked = False
        self.auth_user = None
        self.auth_message = ""
        self.auth_status = "locked"  # locked | unlocked | authenticating

        # Registration state (Phase 1: multi-frame capture)
        self.register_capture = False  # Flag to trigger capture
        self.register_username = ""
        self.register_result = None
        self.register_status = ""
        self.register_collected_faces = []  # Multi-frame face collection
        self.register_min_frames = 15  # Minimum frames for centroid mode

        # Latest frame data for API
        self.latest_frame = None
        self.latest_faces = []
        self.latest_spoof_results = {}

        # Models (loaded once)
        self.config = None
        self.antispoof_model = None
        self.antispoof_transform = None
        self.face_detector = None
        self.face_encoder = None
        self.identity_store = None
        self.attempt_logger = None
        self.registration_mgr = None
        self.auth_mgr = None
        self.decision_engine = None  # Phase 2
        self.blink_challenge = None   # Phase 3
        self.integrity_ok = True       # Phase 5
        self.device = None

        # Session timeout auto-lock
        self.session_timeout_s: int = 300     # default 5 min; overridden from config
        self.last_activity_time: float = 0.0  # epoch when system was last unlocked/used

        # Post-lockout liveness gate
        # Set True when device lockout expires → next auth requires blink challenge
        self.post_lockout: bool = False

        # SSE event queue (shared across all listeners)
        self._sse_listeners: list = []        # list of queue.Queue per subscriber
        self._sse_lock = threading.Lock()

        # Smoothing
        self.smoothing = 20
        self.spoof_histories = {}
        self.image_size = 112

        # Camera thread
        self.camera_thread = None
        self.camera_running = False
        self.current_mode = None  # 'register' or 'unlock'

        # Risk-based active liveness challenge state (Phase 3)
        self.risk_challenges = {}  # client_ip -> BlinkChallenge
        self.liveness_passed = {}  # client_ip -> bool


state = AppState()


def load_config(config_path="config/config_auth.yaml"):
    """Load auth config."""
    full_path = os.path.join(PROJECT_ROOT, config_path)
    with open(full_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg.get('auth', cfg)


def init_models():
    """Load all models once at startup."""
    if state.antispoof_model is not None:
        return  # Already loaded

    print("[Web] Loading models...")
    config = load_config()
    state.config = config

    # Phase 5: File integrity check
    checker = IntegrityChecker(project_root=PROJECT_ROOT)
    passed, report = checker.verify()
    state.integrity_ok = passed
    if not passed:
        print("[Web] ⚠  SYSTEM INTEGRITY COMPROMISED — auth will be blocked")
        logger.critical(f"Integrity check failed: {report['message']}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state.device = device
    print(f"[Web] Device: {device}")

    # Anti-spoof model
    model_name = config.get('antispoof_model', 'attention_cnn')
    checkpoint_path = config.get('antispoof_checkpoint',
                                 'checkpoints/attention_cnn/best_model.pth')
    version = config.get('antispoof_version', 'v2')

    print(f"[Web] Loading anti-spoof: {model_name} ({version})")
    model = get_model(model_name, num_classes=2, pretrained=False, version=version)
    ckpt_full = os.path.join(PROJECT_ROOT, checkpoint_path)
    checkpoint = torch.load(ckpt_full, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    state.antispoof_model = model

    state.image_size = config.get('image_size', 112)
    state.antispoof_transform = get_val_transforms(image_size=state.image_size)
    state.smoothing = config.get('smoothing', 20)

    # Face detector
    detector_type = config.get('detector', 'retinaface')
    print(f"[Web] Loading face detector: {detector_type}")
    state.face_detector = FaceDetector(
        device=str(device).split(':')[0],
        detector_type=detector_type,
    )

    # Face encoder
    print("[Web] Loading face encoder...")
    state.face_encoder = FaceEncoder(device=str(device))

    # Identity store
    state.identity_store = IdentityStore(
        store_path=os.path.join(PROJECT_ROOT, config.get('store_path', 'data/auth/users.enc')),
        key_path=os.path.join(PROJECT_ROOT, config.get('key_path', 'data/auth/auth.key')),
        max_users=config.get('max_users', 5),
    )

    # Attempt logger
    state.attempt_logger = AttemptLogger(
        log_path=os.path.join(PROJECT_ROOT, config.get('log_path', 'data/auth/attempts.csv')),
    )

    # Registration manager
    state.registration_mgr = RegistrationManager(
        face_encoder=state.face_encoder,
        identity_store=state.identity_store,
        spoof_threshold=config.get('spoof_threshold', 0.50),
    )

    # Authentication manager — temporal gate + blur gate + A/B config
    full_config = _load_full_config()           # top-level config (includes 'security')
    sec_cfg  = full_config.get('security', {})
    tc_cfg   = sec_cfg.get('temporal_consistency', {})
    aq_cfg   = sec_cfg.get('auth_quality', {})
    consecutive_required = (
        tc_cfg.get('consecutive_required', 2)
        if tc_cfg.get('enabled', True) else 1
    )
    blur_threshold = aq_cfg.get('blur_threshold', 60.0)

    state.auth_mgr = AuthenticationManager(
        face_encoder=state.face_encoder,
        identity_store=state.identity_store,
        attempt_logger=state.attempt_logger,
        identity_threshold=config.get('identity_threshold', 0.85),
        lockout_attempt_limit=config.get('lockout_attempt_limit', 10),
        lockout_duration=config.get('lockout_duration', 3),
        ab_config=config.get('ab_mode', {}),
        consecutive_required=consecutive_required,
        auth_blur_threshold=blur_threshold,
    )

    # IP Rate Limiter
    rl_cfg = sec_cfg.get('ip_rate_limit', {})
    if rl_cfg.get('enabled', True):
        state.ip_limiter = IPRateLimiter(
            max_fails=rl_cfg.get('max_fails', 5),
            window_s=rl_cfg.get('window_s', 60),
            block_duration_s=rl_cfg.get('block_duration_s', 300),
            auto_expire=rl_cfg.get('auto_expire', True),
        )
        print(f"[Web] IP rate limiter: block after {rl_cfg.get('max_fails',5)} fails "
              f"in {rl_cfg.get('window_s',60)}s")
    else:
        state.ip_limiter = None

    # 2FA PIN store
    pin_cfg = sec_cfg.get('pin_2fa', {})
    state.pin_2fa_enabled = pin_cfg.get('enabled', False)
    state.pin_grace_s     = pin_cfg.get('grace_period_s', 30)
    state.pin_store = PINStore(
        store_path=os.path.join(PROJECT_ROOT, pin_cfg.get('store_path', 'data/auth/pins.enc')),
        key_path=os.path.join(PROJECT_ROOT, pin_cfg.get('key_path', 'data/auth/pins.key')),
    )
    print(f"[Web] 2FA PIN: {'ENABLED' if state.pin_2fa_enabled else 'disabled'}")

    # Phase 2: Risk-based decision engine
    state.decision_engine = RiskDecisionEngine.from_config(config)

    # Phase 3: Blink challenge
    state.blink_challenge = BlinkChallenge.from_config(config)

    # Session timeout
    state.session_timeout_s = full_config.get('security', {}).get('session_timeout_s', 300)
    print(f"[Web] Session timeout: {state.session_timeout_s}s")

    print(f"[Web] Registered users: {state.identity_store.list_users()}")
    print("[Web] All models loaded!\n")


# =============================================================================
# SSE helper
# =============================================================================
def push_event(event_type: str, data: dict):
    """Push a JSON event to all active SSE subscribers."""
    payload = json.dumps({'type': event_type, 'ts': time.time(), **data})
    with state._sse_lock:
        dead = []
        for q in state._sse_listeners:
            try:
                q.put_nowait(payload)
            except Exception:
                dead.append(q)
        for q in dead:
            state._sse_listeners.remove(q)


# =============================================================================
# Video processing
# =============================================================================
def decode_base64_image(base64_str):
    """Decode base64 image string to BGR OpenCV image."""
    try:
        import base64
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        return None


def preprocess_face(face_img):
    """Preprocess face for anti-spoof model (matches face_auth_demo.py)."""
    from PIL import Image as PILImage
    # Apply low-light enhancement adaptively
    face_img = enhance_face_illumination(face_img)
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    pil_image = PILImage.fromarray(face_rgb)
    tensor = state.antispoof_transform(pil_image)
    tensor = tensor.unsqueeze(0).to(state.device)
    return tensor


def predict_antispoof(face_img, face_id=0):
    """Run anti-spoof prediction with smoothing."""
    tensor = preprocess_face(face_img)

    with torch.no_grad():
        output = state.antispoof_model(tensor)
        probs = torch.softmax(output, dim=1)
        spoof_prob = probs[0, 1].item()

    # Smoothing
    if face_id not in state.spoof_histories:
        state.spoof_histories[face_id] = []

    history = state.spoof_histories[face_id]
    history.append(spoof_prob)
    if len(history) > state.smoothing:
        history.pop(0)

    smoothed = sum(history) / len(history)

    threshold = state.config.get('spoof_threshold', 0.50)
    is_spoof = smoothed >= threshold
    label = "SPOOF" if is_spoof else "REAL"
    confidence = (1 - smoothed) if not is_spoof else smoothed

    return {
        'is_spoof': is_spoof,
        'spoof_prob': smoothed,
        'label': label,
        'confidence': confidence,
        'raw_prob': spoof_prob,
    }


# =============================================================================
# Routes
# =============================================================================
@app.route('/')
def home():
    """Home page — locked/unlocked. Auto-logouts admin to prevent session lingering."""
    session.pop('admin_logged_in', None)
    session.pop('admin_username', None)
    return render_template('home.html')


@app.route('/register')
def register_page():
    """Face registration page."""
    return render_template('register.html')


# =============================================================================
# API endpoints
# =============================================================================
@app.route('/api/users', methods=['GET'])
def list_users():
    """List registered users."""
    users = state.identity_store.list_users()
    return jsonify({
        'users': users,
        'count': len(users),
        'max': state.identity_store.max_users,
    })


@app.route('/api/users/add', methods=['POST'])
def add_user():
    """Register a new user with multiple base64 encoded face frames and optionally set a 2FA PIN."""
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    pin = str(data.get('pin', '')).strip()
    images = data.get('images', [])

    if not username:
        return jsonify({'success': False, 'message': 'Username cannot be empty'}), 400

    if state.identity_store.user_count() >= state.identity_store.max_users:
        return jsonify({
            'success': False,
            'message': f'User limit reached ({state.identity_store.max_users})'
        }), 400

    if username in state.identity_store.list_users():
        return jsonify({
            'success': False,
            'message': f'Username "{username}" already exists'
        }), 400

    # Validate PIN format early if 2FA is enabled and a PIN was provided
    if state.pin_2fa_enabled and getattr(state, 'pin_store', None):
        if not pin:
            return jsonify({'success': False, 'message': '2FA is enabled: a 4-8 digit PIN is required'}), 400
        if not pin.isdigit() or not (4 <= len(pin) <= 8):
            return jsonify({'success': False, 'message': 'PIN must be 4–8 digits'}), 400

    if not images or len(images) < 5:
        return jsonify({
            'success': False,
            'message': 'Insufficient face frames provided. Need at least 5 frames to register.'
        }), 400

    collected_faces = []
    det_scale = 0.5

    for idx, base64_img in enumerate(images):
        frame = decode_base64_image(base64_img)
        if frame is None:
            continue

        # Scale down for detection
        small_frame = cv2.resize(frame, (0, 0), fx=det_scale, fy=det_scale)
        small_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        faces_small = state.face_detector.detect_faces(small_rgb)

        if not faces_small:
            # Fallback to full resolution if downscaled fails
            faces_small = state.face_detector.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not faces_small:
                logger.warning(f"No face detected in registration frame {idx}")
                continue
            x, y, w, h = faces_small[0]
        else:
            x, y, w, h = faces_small[0]
            x, y, w, h = int(x / det_scale), int(y / det_scale), int(w / det_scale), int(h / det_scale)

        h_img, w_img = frame.shape[:2]
        fx1 = max(0, x)
        fy1 = max(0, y)
        fw = min(w, w_img - fx1)
        fh = min(h, h_img - fy1)
        if fw < 20 or fh < 20:
            continue
        face_crop = frame[fy1:fy1+fh, fx1:fx1+fw]
        if face_crop.size == 0:
            continue
        face_crop = enhance_face_illumination(face_crop)

        try:
            # Resize face to model input size (matches predict_antispoof inputs)
            face_resized = cv2.resize(face_crop, (state.image_size, state.image_size))
            spoof_result = predict_antispoof(face_resized, face_id=0)
            if not spoof_result.get('is_spoof', True):
                collected_faces.append(face_crop.copy())
            else:
                logger.warning(f"Registration frame {idx} failed liveness check: spoof score={spoof_result.get('spoof_prob', 1.0):.3f}")
        except Exception as e:
            logger.error(f"Error predicting anti-spoof on registration frame {idx}: {e}")

    # Require at least 10 valid real face crops
    min_required = max(5, min(10, state.register_min_frames))
    if len(collected_faces) < min_required:
        return jsonify({
            'success': False,
            'message': f'Registration failed: only {len(collected_faces)}/{len(images)} frames passed liveness verification. Please stay still in a well-lit environment.'
        }), 400

    # Register with multiple frames + adaptive threshold
    adaptive_cfg = state.config.get('adaptive_identity', {})
    result = state.registration_mgr.register_multi(
        username,
        collected_faces,
        adaptive_config=adaptive_cfg if adaptive_cfg.get('enabled', True) else None,
    )

    if result.get('success', False):
        if state.pin_2fa_enabled and getattr(state, 'pin_store', None) and pin:
            pin_success, pin_msg = state.pin_store.set_pin(username, pin)
            if pin_success:
                result['message'] += " (2FA PIN saved)"
            else:
                result['message'] += f" (Warning: {pin_msg})"
        return jsonify(result)
    else:
        return jsonify(result), 400



@app.route('/api/users/remove', methods=['POST'])
def remove_user():
    """Remove a registered user."""
    data = request.get_json()
    username = data.get('username', '').strip()

    if not username:
        return jsonify({'success': False, 'message': 'Username required'}), 400

    ok, msg = state.identity_store.remove_user(username)
    return jsonify({'success': ok, 'message': msg})


@app.route('/api/users/rename', methods=['POST'])
def rename_user():
    """Rename a registered user."""
    data = request.get_json()
    old_name = data.get('old_name', '').strip()
    new_name = data.get('new_name', '').strip()

    if not old_name or not new_name:
        return jsonify({'success': False, 'message': 'Both old and new names required'}), 400

    ok, msg = state.identity_store.rename_user(old_name, new_name)
    return jsonify({'success': ok, 'message': msg})


@app.route('/api/authenticate', methods=['POST'])
def authenticate():
    """Run face authentication against current webcam frame."""
    client_ip = request.remote_addr or '0.0.0.0'

    # ── IP Rate Limit check ───────────────────────────────────────────────
    if getattr(state, 'ip_limiter', None):
        blocked, remaining = state.ip_limiter.is_blocked(client_ip)
        if blocked:
            logger.warning(f"Auth blocked: IP {client_ip} is rate-limited (remaining={remaining})")
            block_msg = (
                'IP blocked — contact admin to unblock.'
                if remaining <= 0 else
                f'Too many failed attempts. Try again in {remaining}s.'
            )
            return jsonify({
                'status': 'ip_blocked',
                'message': block_msg,
                'user': None,
                'confidence': 0.0,
                'lockout_remaining': remaining,
            }), 429

    # Decode frame from client payload
    req_data = request.get_json() or {}
    base64_img = req_data.get('image')
    if not base64_img:
        return jsonify({
            'status': 'error',
            'message': 'No webcam image data provided.',
            'user': None,
            'confidence': 0.0
        }), 400

    frame = decode_base64_image(base64_img)
    if frame is None:
        return jsonify({
            'status': 'error',
            'message': 'Failed to decode webcam image.',
            'user': None,
            'confidence': 0.0
        }), 400

    # ── Post-lockout liveness gate ────────────────────────────────────────────
    # After 5 consecutive failures the device locks. When it unlocks, the
    # next auth attempt MUST pass a blink challenge before face matching resumes.
    if getattr(state, 'post_lockout', False):
        # Lazy-init per-IP challenge instances
        if not hasattr(state, '_post_lockout_challenges'):
            state._post_lockout_challenges = {}

        bc_template = getattr(state, 'blink_challenge', None)

        if bc_template is not None:
            try:
                # Get or create a fresh BlinkChallenge instance for this IP
                if client_ip not in state._post_lockout_challenges:
                    from src.liveness import BlinkChallenge
                    challenge = BlinkChallenge(required_blinks=1, timeout=30.0)
                    challenge.start()
                    state._post_lockout_challenges[client_ip] = challenge

                challenge = state._post_lockout_challenges[client_ip]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = challenge.update(frame_rgb)

                if result is None:
                    # Challenge still in progress — waiting for blink
                    remaining = round(challenge.get_remaining_time(), 1)
                    return jsonify({
                        'status': 'post_lockout_challenge',
                        'message': f'Blink once to verify liveness ({remaining}s remaining).',
                        'user': None,
                        'confidence': 0.0,
                        'lockout_remaining': remaining,
                    }), 200

                if result.get('passed'):
                    # Blink confirmed — clear gate
                    state._post_lockout_challenges.pop(client_ip, None)
                    with state.lock:
                        state.post_lockout = False
                    logger.info(f"Post-lockout blink challenge PASSED for IP {client_ip}")
                else:
                    # Timed out without blink — reset so they can retry
                    state._post_lockout_challenges.pop(client_ip, None)
                    return jsonify({
                        'status': 'post_lockout_challenge',
                        'message': 'Liveness check timed out — please blink when ready.',
                        'user': None,
                        'confidence': 0.0,
                        'lockout_remaining': 0,
                    }), 200

            except Exception as e:
                # MediaPipe not compatible (e.g. mp.solutions missing in v0.10+)
                # Fail-open: log warning and clear the gate so auth is not blocked
                logger.warning(f"BlinkChallenge failed ({e}) — skipping liveness gate")
                state._post_lockout_challenges.pop(client_ip, None)
                with state.lock:
                    state.post_lockout = False
        else:
            # No BlinkChallenge module or no frame — clear flag and proceed
            with state.lock:
                state.post_lockout = False
            if hasattr(state, '_post_lockout_challenges'):
                state._post_lockout_challenges.pop(client_ip, None)

    # ── Standard Risk-based Blink Challenge check (Phase 3) ──────────────────
    if client_ip in state.risk_challenges:
        challenge = state.risk_challenges[client_ip]
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = challenge.update(frame_rgb)

            if result is None:
                # Challenge still in progress — waiting for blink
                remaining = round(challenge.get_remaining_time(), 1)
                return jsonify({
                    'status': 'challenge_in_progress',
                    'message': f"Liveness check in progress: Please blink {challenge.required_blinks} times ({remaining}s remaining).",
                    'user': None,
                    'confidence': 0.0,
                    'blinks_detected': challenge.detector.blink_count,
                    'required_blinks': challenge.required_blinks,
                    'lockout_remaining': remaining,
                }), 200

            if result.get('passed'):
                # Blink challenge passed — set bypass flag and proceed to identity matching
                state.risk_challenges.pop(client_ip, None)
                state.liveness_passed[client_ip] = True
                logger.info(f"Standard risk blink challenge PASSED for IP {client_ip}")
            else:
                # Timed out or failed
                state.risk_challenges.pop(client_ip, None)
                state.liveness_passed.pop(client_ip, None)
                return jsonify({
                    'status': 'challenge_failed',
                    'message': 'Liveness check failed — please try again.',
                    'user': None,
                    'confidence': 0.0,
                }), 200
        except Exception as e:
            logger.warning(f"BlinkChallenge failed during active challenge ({e}) — clearing challenge")
            state.risk_challenges.pop(client_ip, None)

    # Phase 5: Block auth if integrity check failed
    if not state.integrity_ok:
        return jsonify({
            'status': 'integrity_error',
            'message': 'System Integrity Compromised — authentication blocked',
            'user': None,
            'confidence': 0.0,
        }), 403

    # Run face detection synchronously
    faces = []
    try:
        det_scale = 0.5
        small_frame = cv2.resize(frame, (0, 0), fx=det_scale, fy=det_scale)
        small_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        faces_small = state.face_detector.detect_faces(small_rgb)
        faces = [
            (int(x / det_scale), int(y / det_scale), int(w / det_scale), int(h / det_scale))
            for (x, y, w, h) in faces_small
        ]
    except Exception as e:
        logger.error(f"Face detection failed: {e}")

    if len(faces) == 0:
        return jsonify({
            'status': 'no_face',
            'message': 'No face detected. Please look at the camera.',
            'user': None,
            'confidence': 0.0,
        })

    # Run anti-spoofing synchronously on the first face
    spoof_results = {}
    x, y, w, h = faces[0]
    h_img, w_img = frame.shape[:2]
    fx1 = max(0, x)
    fy1 = max(0, y)
    fw = min(w, w_img - fx1)
    fh = min(h, h_img - fy1)

    if fw < 20 or fh < 20:
        return jsonify({
            'status': 'no_face',
            'message': 'Face is too small or out of bounds.',
            'user': None,
            'confidence': 0.0,
        })

    face_crop = frame[fy1:fy1+fh, fx1:fx1+fw]
    face_crop = enhance_face_illumination(face_crop)
    try:
        face_resized = cv2.resize(face_crop, (state.image_size, state.image_size))
        spoof_results[0] = predict_antispoof(face_resized, face_id=0)
    except Exception as e:
        logger.error(f"Anti-spoofing processing failed: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Anti-spoofing processing failed.',
            'user': None,
            'confidence': 0.0,
        })

    # Phase 2: Run risk-based decision engine
    liveness_already_passed = state.liveness_passed.get(client_ip, False)

    if liveness_already_passed:
        # Liveness verification already completed, proceed to matching
        auth_result = state.auth_mgr.authenticate(face_crop, spoof_results[0])
    else:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        risk_result = state.decision_engine.evaluate(
            spoof_result=spoof_results[0],
            face_bbox=faces[0],
            face_img=face_crop,
            frame_gray=frame_gray,
        )

        if risk_result['enabled']:
            band = risk_result['band']
            if band == RiskDecisionEngine.BAND_PASS:
                # Proceed to identity authentication
                auth_result = state.auth_mgr.authenticate(face_crop, spoof_results[0])
            elif band == RiskDecisionEngine.BAND_CHALLENGE:
                # Initialize active liveness challenge
                from src.liveness import BlinkChallenge
                liveness_cfg = state.config.get('liveness', {})
                challenge = BlinkChallenge(
                    required_blinks=liveness_cfg.get('blink_count', 2),
                    timeout=liveness_cfg.get('challenge_timeout', 5.0),
                    ear_threshold=liveness_cfg.get('ear_threshold', 0.21),
                    debounce_frames=liveness_cfg.get('debounce_frames', 3)
                )
                challenge.start()
                state.risk_challenges[client_ip] = challenge
                
                remaining = round(challenge.get_remaining_time(), 1)
                auth_result = {
                    'status': 'challenge_in_progress',
                    'message': f"Liveness check in progress: Please blink {challenge.required_blinks} times ({remaining}s remaining).",
                    'user': None,
                    'confidence': 0.0,
                    'blinks_detected': 0,
                    'required_blinks': challenge.required_blinks,
                    'lockout_remaining': remaining,
                }
            else:
                # FAIL band
                auth_result = {
                    'status': 'spoof_denied',
                    'message': f"Risk check failed (score: {risk_result['score']:.2f})",
                    'user': None,
                    'confidence': 0.0,
                    'risk_score': risk_result['score'],
                }
        else:
            # Fallback: original mode
            auth_result = state.auth_mgr.authenticate(face_crop, spoof_results[0])

    # ── IP Rate Limiter: record outcome ──────────────────────────────────
    if getattr(state, 'ip_limiter', None):
        status = auth_result.get('status', '')
        if status == 'access_granted':
            state.ip_limiter.record_success(client_ip)
        elif status not in ('scanning', 'processing', 'no_face', 'integrity_error'):
            blocked_now, _ = state.ip_limiter.record_failure(client_ip)
            if blocked_now:
                logger.warning(f"IP {client_ip} has been blocked after repeated failures")

    # Phase 4: Issue JWT on successful authentication
    if auth_result.get('status') == 'access_granted' and JWT_AVAILABLE:
        username = auth_result['user']
        access_token = create_access_token(identity=username)
        refresh_token = create_refresh_token(identity=username)
        auth_result['access_token'] = access_token
        auth_result['refresh_token'] = refresh_token
        logger.info(f"JWT issued for user: {username}")

    with state.lock:
        status = auth_result.get('status', '')
        if status == 'access_granted':
            username = auth_result['user']
            state.post_lockout = False  # clear post-lockout gate
            # 2FA: if PIN enabled and user has a PIN, require it next
            if (getattr(state, 'pin_2fa_enabled', False)
                    and getattr(state, 'pin_store', None)
                    and state.pin_store.has_pin(username)):
                state.is_unlocked = False
                state.auth_user = username
                state.auth_status = 'pending_pin'
                auth_result['status']  = 'pending_pin'
                auth_result['message'] = 'Face verified — enter PIN to unlock'
            else:
                state.is_unlocked = True
                state.auth_user = username
                state.auth_status = 'unlocked'
                state.last_activity_time = time.time()  # start session timer
        elif status == 'locked_out':
            state.post_lockout = True  # require blink when lockout expires
            state.auth_status = 'locked'
        else:
            state.auth_status = 'locked'
        state.auth_message = auth_result.get('message', '')

    # Push live SSE event for admin dashboard
    push_event('auth_attempt', {
        'status': auth_result.get('status', ''),
        'user': auth_result.get('user'),
        'confidence': round(float(auth_result.get('confidence', 0.0)), 4),
        'ip': client_ip,
    })

    # Clear liveness pass if they reached a final decision
    status = auth_result.get('status', '')
    if status in ('access_granted', 'pending_pin', 'spoof_denied', 'unauthorized', 'locked_out', 'ip_blocked', 'challenge_failed'):
        with state.lock:
            state.liveness_passed.pop(client_ip, None)

    return jsonify(auth_result)


@app.route('/api/auth_status', methods=['GET'])
@jwt_required(optional=True)
def auth_status():
    """Get current authentication state. JWT optional for backward compat."""
    jwt_user = get_jwt_identity() if JWT_AVAILABLE else None
    with state.lock:
        return jsonify({
            'status': state.auth_status,
            'user': state.auth_user,
            'message': state.auth_message,
            'is_unlocked': state.is_unlocked,
            'jwt_user': jwt_user,
        })


@app.route('/api/lock', methods=['POST'])
def lock_system():
    """Re-lock the system."""
    client_ip = request.remote_addr or '0.0.0.0'
    with state.lock:
        state.is_unlocked = False
        state.auth_user = None
        state.auth_status = 'locked'
        state.auth_message = ''
        state.risk_challenges.pop(client_ip, None)
        state.liveness_passed.pop(client_ip, None)
    push_event('lock', {'user': state.auth_user, 'reason': 'manual'})
    return jsonify({'success': True, 'message': 'System locked'})


# =============================================================================
# 2FA PIN Routes
# =============================================================================

@app.route('/api/pin/verify', methods=['POST'])
def verify_pin():
    """Verify 2FA PIN after successful face match."""
    data = request.json or {}
    pin = data.get('pin', '').strip()
    
    with state.lock:
        # Only allow PIN entry if we are in the 'pending_pin' state
        if state.auth_status != 'pending_pin' or not state.auth_user:
            return jsonify({'success': False, 'message': 'PIN not currently required'}), 400

        username = state.auth_user
        
        # Verify the PIN against the encrypted store
        if state.pin_store and state.pin_store.verify_pin(username, pin):
            state.is_unlocked = True
            state.auth_status = 'unlocked'
            state.last_activity_time = time.time()  # Start session timer
            
            logger.info(f"2FA PIN verified successfully for user {username}")
            push_event('auth_attempt', {
                'status': 'access_granted',
                'user': username,
                'confidence': 1.0,
                'ip': request.remote_addr,
            })
            return jsonify({'success': True, 'message': 'System unlocked'})
        else:
            # Failed PIN attempt
            logger.warning(f"Failed 2FA PIN attempt for user {username}")
            push_event('auth_attempt', {
                'status': 'unauthorized',
                'user': username,
                'confidence': 0.0,
                'ip': request.remote_addr,
            })
            
            # Lock the system back down on failure to prevent brute forcing
            state.is_unlocked = False
            state.auth_status = 'locked'
            state.auth_user = None
            return jsonify({'success': False, 'message': 'Incorrect PIN'}), 401


@app.route('/api/pin/set', methods=['POST'])
def set_pin():
    """Set or update the PIN for the CURRENTLY AUTHENTICATED user."""
    data = request.json or {}
    new_pin = str(data.get('pin', '')).strip()
    
    with state.lock:
        # Security: Can only set a PIN if the system is currently unlocked!
        if not state.is_unlocked or not state.auth_user:
            return jsonify({'success': False, 'message': 'Authentication required to set PIN'}), 401
            
        if not state.pin_2fa_enabled or not getattr(state, 'pin_store', None):
            return jsonify({'success': False, 'message': '2FA PIN feature is disabled'}), 400
            
        username = state.auth_user
        
        success, msg = state.pin_store.set_pin(username, new_pin)
        if success:
            logger.info(f"User {username} updated their 2FA PIN")
            return jsonify({'success': True, 'message': msg})
        else:
            return jsonify({'success': False, 'message': msg}), 400


# =============================================================================
# Session Timeout
# =============================================================================

@app.route('/api/session_status', methods=['GET'])
def session_status():
    """Return remaining session time and auto-lock if timeout reached."""
    with state.lock:
        is_unlocked = state.is_unlocked
        last_activity = state.last_activity_time
        timeout = state.session_timeout_s
        user = state.auth_user

    if not is_unlocked or last_activity == 0:
        return jsonify({'is_unlocked': False, 'remaining_s': 0, 'timed_out': False})

    elapsed = time.time() - last_activity
    remaining = max(0, int(timeout - elapsed))

    if remaining == 0:
        # Auto-lock
        with state.lock:
            state.is_unlocked  = False
            state.auth_user    = None
            state.auth_status  = 'locked'
            state.auth_message = 'Session timed out'
            state.last_activity_time = 0.0
        push_event('lock', {'user': user, 'reason': 'session_timeout'})
        logger.info(f"Session timed out for user: {user}")
        return jsonify({'is_unlocked': False, 'remaining_s': 0, 'timed_out': True, 'user': user})

    return jsonify({'is_unlocked': True, 'remaining_s': remaining, 'timed_out': False, 'user': user})


@app.route('/api/session_keepalive', methods=['POST'])
def session_keepalive():
    """Reset session timer (call on user interaction)."""
    with state.lock:
        if state.is_unlocked:
            state.last_activity_time = time.time()
    return jsonify({'success': True})


# =============================================================================
# SSE Live Event Feed
# =============================================================================

@app.route('/api/events')
def sse_events():
    """Server-Sent Events stream for admin live dashboard."""
    if not _is_admin():
        return jsonify({'error': 'Admin required'}), 403

    q = queue.Queue(maxsize=50)
    with state._sse_lock:
        state._sse_listeners.append(q)

    def generate():
        # Send connected ping
        yield f"data: {json.dumps({'type': 'connected', 'ts': time.time()})}\n\n"
        try:
            while True:
                try:
                    payload = q.get(timeout=25)
                    yield f"data: {payload}\n\n"
                except queue.Empty:
                    # keepalive ping every 25s
                    yield f"data: {json.dumps({'type': 'ping', 'ts': time.time()})}\n\n"
        except GeneratorExit:
            pass
        finally:
            with state._sse_lock:
                if q in state._sse_listeners:
                    state._sse_listeners.remove(q)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


# =============================================================================
# Security Policy Management
# =============================================================================

@app.route('/api/admin/policy', methods=['GET'])
def get_security_policy():
    """Retrieve the current security policy parameters. Admin only."""
    if not _is_admin():
        return jsonify({'error': 'Admin authentication required'}), 403
    
    try:
        cfg = _load_full_config()
        # Filter out sensitive admin passwords for safety
        policy = {
            'auth': cfg.get('auth', {}),
            'security': cfg.get('security', {})
        }
        return jsonify({'success': True, 'policy': policy})
    except Exception as e:
        logger.error(f"Failed to read security policy: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/admin/policy/update', methods=['POST'])
def update_security_policy():
    """Update the security policy parameters. Admin only."""
    if not _is_admin():
        return jsonify({'error': 'Admin authentication required'}), 403
    
    data = request.get_json() or {}
    new_auth = data.get('auth', {})
    new_security = data.get('security', {})
    
    try:
        cfg_path = os.path.join(PROJECT_ROOT, 'config', 'config_auth.yaml')
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f) or {}
            
        # Update config fields safely
        if 'auth' not in cfg:
            cfg['auth'] = {}
        if 'security' not in cfg:
            cfg['security'] = {}
            
        # Update auth settings
        for k, v in new_auth.items():
            if isinstance(v, dict) and k in cfg['auth'] and isinstance(cfg['auth'][k], dict):
                cfg['auth'][k].update(v)
            else:
                cfg['auth'][k] = v
                
        # Update security settings
        for k, v in new_security.items():
            if isinstance(v, dict) and k in cfg['security'] and isinstance(cfg['security'][k], dict):
                cfg['security'][k].update(v)
            else:
                cfg['security'][k] = v
                
        with open(cfg_path, 'w') as f:
            yaml.safe_dump(cfg, f, default_flow_style=False)
            
        # Reload state in real-time
        auth_cfg = cfg.get('auth', {})
        state.config = auth_cfg
        
        # Reload Session Timeout
        state.session_timeout_s = cfg.get('security', {}).get('session_timeout_s', 30)
        
        # Reload 2FA PIN settings
        pin_cfg = cfg.get('security', {}).get('pin_2fa', {})
        state.pin_2fa_enabled = pin_cfg.get('enabled', False)
        state.pin_grace_s = pin_cfg.get('grace_period_s', 30)
        
        # Re-init RiskDecisionEngine
        from src.auth import RiskDecisionEngine
        state.decision_engine = RiskDecisionEngine.from_config(auth_cfg)
        
        # Re-init BlinkChallenge
        from src.liveness import BlinkChallenge
        state.blink_challenge = BlinkChallenge.from_config(auth_cfg)
        
        # Re-init AuthenticationManager
        sec_cfg = cfg.get('security', {})
        tc_cfg = sec_cfg.get('temporal_consistency', {})
        aq_cfg = sec_cfg.get('auth_quality', {})
        consecutive_required = tc_cfg.get('consecutive_required', 2) if tc_cfg.get('enabled', True) else 1
        blur_threshold = aq_cfg.get('blur_threshold', 60.0)
        
        if state.auth_mgr:
            state.auth_mgr.identity_threshold = auth_cfg.get('identity_threshold', 0.85)
            state.auth_mgr.lockout_attempt_limit = auth_cfg.get('lockout_attempt_limit', 10)
            state.auth_mgr.lockout_duration = auth_cfg.get('lockout_duration', 3)
            state.auth_mgr.ab_config = auth_cfg.get('ab_mode', {})
            state.auth_mgr.consecutive_required = consecutive_required
            state.auth_mgr.auth_blur_threshold = blur_threshold
            
        # Re-init IP Rate Limiter
        rl_cfg = sec_cfg.get('ip_rate_limit', {})
        if rl_cfg.get('enabled', True):
            from src.auth import IPRateLimiter
            state.ip_limiter = IPRateLimiter(
                max_fails=rl_cfg.get('max_fails', 5),
                window_s=rl_cfg.get('window_s', 60),
                block_duration_s=rl_cfg.get('block_duration_s', 300),
                auto_expire=rl_cfg.get('auto_expire', True)
            )
        else:
            state.ip_limiter = None
            
        logger.info(f"Security Policy updated in real-time by admin: {session.get('admin_username')}")
        return jsonify({'success': True, 'message': 'Security Policy updated and reloaded successfully'})
    except Exception as e:
        logger.error(f"Failed to update security policy: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


# =============================================================================
# Data Export
# =============================================================================

@app.route('/api/admin/export_log', methods=['GET'])
def admin_export_log():
    """Download attempt log as CSV. Admin only."""
    if not _is_admin():
        return jsonify({'error': 'Admin authentication required'}), 403
    from flask import send_file
    log_path = os.path.join(PROJECT_ROOT, 'data/auth/attempts.csv')
    if not os.path.exists(log_path):
        return jsonify({'error': 'Log file not found'}), 404
    logger.info(f"Log exported by admin: {session.get('admin_username', '?')}")
    return send_file(log_path, as_attachment=True,
                     download_name='auth_attempts.csv',
                     mimetype='text/csv')


# =============================================================================

@app.route('/api/admin/blocked_ips', methods=['GET'])
def admin_blocked_ips():
    """List currently blocked IPs. Admin only."""
    if not _is_admin():
        return jsonify({'error': 'Admin authentication required'}), 403
    limiter = getattr(state, 'ip_limiter', None)
    blocked = limiter.get_blocked_ips() if limiter else []
    stats   = limiter.get_stats()       if limiter else {}
    return jsonify({'blocked_ips': blocked, 'stats': stats})


@app.route('/api/admin/unblock_ip', methods=['POST'])
def admin_unblock_ip():
    """Manually unblock an IP address. Admin only."""
    if not _is_admin():
        return jsonify({'error': 'Admin authentication required'}), 403
    ip = (request.get_json() or {}).get('ip', '').strip()
    if not ip:
        return jsonify({'success': False, 'message': 'IP address required'}), 400
    limiter = getattr(state, 'ip_limiter', None)
    if limiter:
        ok = limiter.unblock(ip)
        logger.info(f"Admin unblocked IP: {ip} by {session.get('admin_username', '?')}")
        return jsonify({'success': ok, 'message': f'IP {ip} unblocked' if ok else 'IP was not blocked'})
    return jsonify({'success': False, 'message': 'IP limiter not enabled'})


@app.route('/api/admin/reset_all_blocks', methods=['POST'])
def admin_reset_all_blocks():
    """Clear all IP blocks. Admin only."""
    if not _is_admin():
        return jsonify({'error': 'Admin authentication required'}), 403
    limiter = getattr(state, 'ip_limiter', None)
    if limiter:
        blocked = limiter.get_blocked_ips()
        count = 0
        for entry in blocked:
            if limiter.unblock(entry['ip']):
                count += 1
        logger.info(f"Admin reset all IP blocks ({count} IPs) by {session.get('admin_username','?')}")
        return jsonify({'success': True, 'message': f'Cleared {count} blocked IPs'})
    return jsonify({'success': False, 'message': 'IP limiter not enabled'})


# =============================================================================
# Security: 2FA PIN Management
# =============================================================================

@app.route('/api/pin/set', methods=['POST'])
@jwt_required(optional=True)
def api_set_pin():
    """Set or update a user's PIN. JWT or matching username required."""
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    pin      = data.get('pin', '').strip()
    if not username or not pin:
        return jsonify({'success': False, 'message': 'Username and PIN required'}), 400
    # Must be the authenticated user or admin
    jwt_user = get_jwt_identity() if JWT_AVAILABLE else None
    with state.lock:
        current_user = state.auth_user
    jwt_match = (str(jwt_user).lower() == username.lower()) if jwt_user else False
    curr_match = (str(current_user).lower() == username.lower()) if current_user else False
    if not _is_admin() and not jwt_match and not curr_match:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    pin_store = getattr(state, 'pin_store', None)
    if not pin_store:
        return jsonify({'success': False, 'message': 'PIN store not initialised'}), 500
    ok, msg = pin_store.set_pin(username, pin)
    return jsonify({'success': ok, 'message': msg})


@app.route('/api/pin/verify', methods=['POST'])
def api_verify_pin():
    """Verify PIN during 2FA flow. Unlocks system if correct."""
    data = request.get_json() or {}
    pin  = data.get('pin', '').strip()
    with state.lock:
        username = state.auth_user
        pending  = state.auth_status == 'pending_pin'
    if not pending or not username:
        return jsonify({'success': False, 'message': 'No pending PIN verification'}), 400
    pin_store = getattr(state, 'pin_store', None)
    if not pin_store:
        return jsonify({'success': False, 'message': 'PIN store not initialised'}), 500
    if pin_store.verify_pin(username, pin):
        with state.lock:
            state.is_unlocked = True
            state.auth_status = 'unlocked'
            state.auth_message = f'Access Granted: {username}'
        logger.info(f"2FA PIN verified for user: {username}")
        return jsonify({'success': True, 'message': f'Access Granted: {username}', 'user': username})
    else:
        # Record IP failure for wrong PIN
        client_ip = request.remote_addr or '0.0.0.0'
        if getattr(state, 'ip_limiter', None):
            state.ip_limiter.record_failure(client_ip)
        logger.warning(f"Wrong PIN for user: {username} from {client_ip}")
        return jsonify({'success': False, 'message': 'Incorrect PIN'}), 401


@app.route('/api/pin/status/<username>', methods=['GET'])
def api_pin_status(username):
    """Check if a user has a PIN set."""
    pin_store = getattr(state, 'pin_store', None)
    has_pin = pin_store.has_pin(username) if pin_store else False
    return jsonify({'username': username, 'has_pin': has_pin,
                    'pin_2fa_enabled': getattr(state, 'pin_2fa_enabled', False)})


@app.route('/api/admin/reset_pin', methods=['POST'])
def admin_reset_pin():
    """Admin: remove a user's PIN."""
    if not _is_admin():
        return jsonify({'error': 'Admin authentication required'}), 403
    username = (request.get_json() or {}).get('username', '').strip()
    pin_store = getattr(state, 'pin_store', None)
    if pin_store and username:
        ok = pin_store.remove_pin(username)
        return jsonify({'success': ok, 'message': f"PIN {'removed' if ok else 'not found'} for {username}"})
    return jsonify({'success': False, 'message': 'Missing username or PIN store'}), 400


# =============================================================================
# Data: Identity Export / Import
# =============================================================================

@app.route('/api/admin/export_identities', methods=['GET'])
def admin_export_identities():
    """Export identity store as encrypted file download. Admin only."""
    if not _is_admin():
        return jsonify({'error': 'Admin authentication required'}), 403
    store_path_full = os.path.join(PROJECT_ROOT, 'data/auth/users.enc')
    if not os.path.exists(store_path_full):
        return jsonify({'error': 'Identity store file not found'}), 404
    from flask import send_file
    logger.info(f"Identity store exported by admin: {session.get('admin_username','?')}")
    return send_file(store_path_full, as_attachment=True,
                     download_name='users_backup.enc',
                     mimetype='application/octet-stream')


# =============================================================================
# Phase 4: JWT Token Management Endpoints
# =============================================================================
@app.route('/api/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh_token():
    """Refresh an access token using a valid refresh token."""
    current_user = get_jwt_identity()
    new_access_token = create_access_token(identity=current_user)
    logger.info(f"Token refreshed for user: {current_user}")
    return jsonify({
        'access_token': new_access_token,
        'user': current_user,
    })


@app.route('/api/logout', methods=['POST'])
@jwt_required(optional=True)
def logout():
    """Logout by blacklisting the current JWT."""
    if JWT_AVAILABLE:
        try:
            jti = get_jwt().get('jti', '')
            if jti:
                _jwt_blacklist.add(jti)
                user = get_jwt_identity()
                logger.info(f"Token blacklisted for user: {user}, jti: {jti}")
        except Exception:
            pass

    with state.lock:
        state.is_unlocked = False
        state.auth_user = None
        state.auth_status = 'locked'
        state.auth_message = ''

    return jsonify({'success': True, 'message': 'Logged out successfully'})


@app.route('/api/blink_challenge', methods=['POST'])
def blink_challenge():
    """
    Phase 3: Run blink challenge for active liveness.
    Captures frames for up to the challenge timeout and checks for blinks.
    """
    state.blink_challenge.start()

    # Process frames until challenge completes or times out
    while state.blink_challenge.is_active():
        with state.lock:
            frame = state.latest_frame

        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            outcome = state.blink_challenge.update(frame_rgb)
            if outcome is not None:
                if outcome['passed']:
                    # Blink passed — now run identity auth
                    with state.lock:
                        faces = list(state.latest_faces)
                        spoof_results = dict(state.latest_spoof_results)
                        auth_frame = state.latest_frame

                    if auth_frame is not None and len(faces) > 0 and 0 in spoof_results:
                        x, y, w, h = faces[0]
                        h_img, w_img = auth_frame.shape[:2]
                        fx1, fy1 = max(0, x), max(0, y)
                        fw = min(w, w_img - fx1)
                        fh = min(h, h_img - fy1)
                        face_crop = auth_frame[fy1:fy1+fh, fx1:fx1+fw]
                        face_crop = enhance_face_illumination(face_crop)

                        auth_result = state.auth_mgr.authenticate(
                            face_crop, spoof_results[0]
                        )
                        with state.lock:
                            if auth_result.get('status') == 'access_granted':
                                state.is_unlocked = True
                                state.auth_user = auth_result['user']
                                state.auth_status = 'unlocked'
                            state.auth_message = auth_result.get('message', '')
                        return jsonify(auth_result)

                return jsonify({
                    'status': 'challenge_failed',
                    'message': outcome['message'],
                    'user': None,
                    'confidence': 0.0,
                })

        time.sleep(0.05)

    # Timeout
    return jsonify({
        'status': 'challenge_failed',
        'message': 'Blink challenge timed out',
        'user': None,
        'confidence': 0.0,
    })




# =============================================================================
# Admin Account & Role-Based Dashboards
# =============================================================================

def _is_admin():
    """Check if the current session is authenticated as admin."""
    return session.get('admin_logged_in') is True


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Admin login page — session-based (separate from face auth)."""
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        # Re-read credentials from disk each time so password changes
        # take effect without a server restart.
        try:
            live_cfg = _load_full_config()
            live_admin = live_cfg.get('admin', {})
            valid_user = str(live_admin.get('username', ADMIN_USERNAME))
            valid_pass = str(live_admin.get('password', ADMIN_PASSWORD))
        except Exception:
            valid_user, valid_pass = str(ADMIN_USERNAME), str(ADMIN_PASSWORD)

        if username == valid_user and password == valid_pass:
            session.permanent = False  # Drop session when browser closes
            session['admin_logged_in'] = True
            session['admin_username'] = username
            logger.info(f"Admin login successful: {username}")
            return redirect(url_for('admin_dashboard'))
        else:
            error = 'Invalid username or password'
            logger.warning(f"Failed admin login attempt for username: '{username}'")
    return render_template('admin_login.html', error=error)


@app.route('/admin/logout')
def admin_logout():
    """Log out from admin session."""
    session.clear()
    logger.info("Admin logged out")
    return redirect(url_for('admin_login'))


@app.route('/admin/dashboard')
def admin_dashboard():
    """Full admin analytics dashboard — requires admin login."""
    if not _is_admin():
        return redirect(url_for('admin_login'))
    logger.info(f"Admin dashboard accessed by: {session.get('admin_username')}")
    registered = state.identity_store.list_users() if state.identity_store else []
    return render_template('admin_dashboard.html', registered_users=registered)


@app.route('/api/admin/metrics', methods=['GET'])
def api_admin_metrics():
    """
    Full admin metrics — all users, all stats, A/B comparison.
    Requires admin session.
    """
    if not _is_admin():
        return jsonify({'error': 'Admin authentication required'}), 403

    # Reuse dashboard_metrics computation (force re-compute, no cache check)
    global _dashboard_cache
    _dashboard_cache['expires'] = 0  # Invalidate cache to get fresh data
    return dashboard_metrics()


@app.route('/user/<username>/dashboard')
def user_dashboard(username):
    """
    Per-user personal dashboard — only shows that user's own data.
    Accessible after face authentication (any authenticated user).
    """
    # Must be face-authenticated and must be viewing own data
    with state.lock:
        current_user = state.auth_user
        is_unlocked  = state.is_unlocked

    if not is_unlocked or not current_user:
        return redirect('/')            # Not authenticated at all → home

    # Users can only see their own dashboard (admins can see anyone's)
    # Use case-insensitive comparison but normalize for lookup
    if not _is_admin():
        if not current_user or current_user.lower() != username.lower():
            return redirect(f'/user/{current_user}/dashboard') if current_user else redirect('/')

    if state.identity_store:
        registered_users = [u.lower() for u in state.identity_store.list_users()]
        if username.lower() not in registered_users:
            return render_template('user_dashboard.html',
                                   username=username, error="User not found")

    logger.info(f"User dashboard accessed: {username} (by: {current_user})")
    return render_template('user_dashboard.html', username=username, error=None)


import math
import numpy as np

def sanitize_for_json(obj):
    """Recursively converts numpy types to native python for safe JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        val = float(obj)
        return None if math.isnan(val) or math.isinf(val) else val
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    return obj

@app.route('/api/user/<username>/metrics', methods=['GET'])
def api_user_metrics(username):
    """
    Per-user metrics — only that user's auth history + threshold info.
    Accessible to the authenticated face-auth user or admin.
    """
    with state.lock:
        current_user = state.auth_user
        is_unlocked  = state.is_unlocked

    username = username.strip()
    # Cross-check permissions (case-insensitive)
    if not _is_admin():
        if not current_user or current_user.lower() != username.lower():
            logger.warning(f"Access denied: current_user '{current_user}' tried to access dashboard for '{username}'")
            return jsonify({'error': 'Access denied to another user\'s data'}), 403

    try:
        import pandas as pd
        log_path = state.config.get('log_path', 'data/auth/attempts.csv')
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            # Ensure critical columns are numeric or strings and cleaned
            df['matched_user'] = df['matched_user'].astype(str).str.strip()
            df['identity_confidence'] = pd.to_numeric(df['identity_confidence'], errors='coerce').fillna(0.0)
            df['_match'] = df['identity_match'].astype(str).str.lower() == 'true'
            
            # Case-insensitive filter
            user_df = df[df['matched_user'].str.lower() == username.lower()]
            total_attempts = int(len(df)) # Calculate here so it's global
            logger.debug(f"Metrics for {username}: total_system={total_attempts}, user_found={len(user_df)}")
        else:
            df = user_df = None
            total_attempts = 0
    except Exception as e:
        logger.warning(f"User metrics CSV error for {username}: {e}")
        df = user_df = None
        total_attempts = 0

    # Auth history stats
    if user_df is not None and len(user_df) > 0:
        # total_attempts is already set above
        user_attempts   = int(len(user_df))
        user_successes  = int(user_df['_match'].sum())
        
        # Guard against NaN if max/min is empty
        conf_col = df['identity_confidence'] # already numeric
        avg_c = user_df['identity_confidence'].mean()
        min_c = user_df['identity_confidence'].min()
        max_c = user_df['identity_confidence'].max()
        
        avg_confidence  = float(avg_c) if pd.notna(avg_c) else 0.0
        min_confidence  = float(min_c) if pd.notna(min_c) else 0.0
        max_confidence  = float(max_c) if pd.notna(max_c) else 0.0
        
        recent_history  = user_df.tail(20)[
            ['timestamp', '_match', 'identity_confidence', 'spoof_label']
        ].copy()
        
        # Convert columns to purely native python types for safe jsonify
        recent_list = []
        for _, row in recent_history.iterrows():
            recent_list.append({
                'timestamp': str(row['timestamp']),
                '_match': bool(row['_match']),
                'identity_confidence': float(row['identity_confidence']) if pd.notna(row['identity_confidence']) else 0.0,
                'spoof_label': str(row['spoof_label']) if pd.notna(row['spoof_label']) else ''
            })
    else:
        # total_attempts is already set above or 0
        user_attempts = user_successes = 0
        avg_confidence = min_confidence = max_confidence = 0.0
        recent_list = []

    # Identity store info (case-insensitive find)
    user_info = {}
    matched_uname = username
    if state.identity_store:
        # First try exact match
        info = state.identity_store.get_user_info(username)
        if not info:
            # Fallback: case-insensitive search
            all_users = state.identity_store.list_users()
            for u in all_users:
                if u.lower() == username.lower():
                    matched_uname = u
                    info = state.identity_store.get_user_info(u)
                    break
        if info:
            user_info = info
            # Update username to the canonical one from the store
            username = matched_uname

    raw_metrics = {
        'username': username,
        'total_system_attempts': total_attempts,
        'user_attempts': user_attempts,
        'user_successes': user_successes,
        'user_success_rate': round(user_successes / user_attempts * 100, 1) if user_attempts else 0,
        'avg_confidence': round(avg_confidence, 4),
        'min_confidence': round(min_confidence, 4),
        'max_confidence': round(max_confidence, 4),
        'recent_history': recent_list,
        'adaptive_threshold': user_info.get('adaptive_threshold'),
        'embedding_count': user_info.get('embedding_count', 0),
        'stats': user_info.get('stats', {}),
        'centroid_norm': user_info.get('centroid_norm'),
        'risk_distribution': {
            'PASS': int((user_df['identity_confidence'] >= 0.85).sum()) if user_df is not None else 0,
            'CHALLENGE': int(((user_df['identity_confidence'] >= 0.75) & (user_df['identity_confidence'] < 0.85)).sum()) if user_df is not None else 0,
            'FAIL': int((user_df['identity_confidence'] < 0.75).sum()) if user_df is not None else 0,
        } if user_df is not None else {'PASS':0, 'CHALLENGE':0, 'FAIL':0}
    }
    
    return jsonify(sanitize_for_json(raw_metrics))


# =============================================================================
# Phase 2: Metrics Dashboard
# =============================================================================
_dashboard_cache: dict = {'data': None, 'expires': 0.0}


@app.route('/dashboard')
@jwt_required(optional=True)
def dashboard():
    """Analytics dashboard — shows auth metrics and per-user stats."""
    jwt_user = get_jwt_identity() if JWT_AVAILABLE else None
    logger.info(f"Dashboard accessed by JWT user: {jwt_user or 'unauthenticated'}")
    return render_template('dashboard.html')


@app.route('/api/dashboard_metrics', methods=['GET'])
@jwt_required(optional=True)
def dashboard_metrics():
    """
    Return aggregated authentication metrics for the dashboard.
    Results are cached for 10 seconds to avoid repeated CSV parsing.
    """
    global _dashboard_cache

    now = time.time()
    if _dashboard_cache['data'] and now < _dashboard_cache['expires']:
        return jsonify(_dashboard_cache['data'])

    logger.info("Dashboard metrics requested — recomputing")

    try:
        import pandas as pd
        log_path = state.config.get('log_path', 'data/auth/attempts.csv')
        if not os.path.exists(log_path):
            df = pd.DataFrame(columns=[
                'timestamp', 'spoof_label', 'spoof_confidence',
                'identity_match', 'identity_confidence', 'matched_user'
            ])
        else:
            df = pd.read_csv(log_path)
    except Exception as e:
        logger.warning(f"Failed to read attempts CSV: {e}")
        df = None

    # ── Attempt statistics ─────────────────────────────────────────────────
    if df is not None and len(df) > 0:
        total = len(df)
        spoof_count = int((df['spoof_label'] == 'SPOOF').sum())
        # identity_match column may be stored as string 'True'/'False'
        df['_match'] = df['identity_match'].astype(str).str.lower() == 'true'
        success_count = int(df['_match'].sum())
        fail_count = total - success_count
        spoof_rate = round(spoof_count / total * 100, 1) if total else 0
        success_rate = round(success_count / total * 100, 1) if total else 0

        # Per-user similarity averages
        per_user_sim = {}
        real_attempts = df[df['spoof_label'] == 'REAL']
        if len(real_attempts) > 0:
            # Group by normalized (stripped) username to avoid duplicates like 'Nihas' vs 'Nihas '
            for user, grp in real_attempts.groupby(real_attempts['matched_user'].astype(str).str.strip()):
                if user and str(user).lower() != 'nan':
                    avg_sim = float(grp['identity_confidence'].astype(float).mean())
                    per_user_sim[str(user)] = round(avg_sim, 4)

        # Attempts over time (last 20 entries for chart)
        recent = df.tail(20)[['timestamp', '_match', 'spoof_label']].copy()
        recent['timestamp'] = recent['timestamp'].astype(str)
        recent_list = recent.to_dict(orient='records')
    else:
        total = success_count = fail_count = spoof_count = 0
        spoof_rate = success_rate = 0
        per_user_sim = {}
        recent_list = []

    # ── Per-user adaptive thresholds from identity store ─────────────────
    per_user_thresholds = {}
    if state.identity_store:
        for uname in state.identity_store.list_users():
            info = state.identity_store.get_user_info(uname)
            if info:
                has_pin = False
                if getattr(state, 'pin_store', None):
                    has_pin = state.pin_store.has_pin(uname)

                per_user_thresholds[uname] = {
                    'adaptive_threshold': info.get('adaptive_threshold'),
                    'embedding_count': info.get('embedding_count', 0),
                    'stats': info.get('stats', {}),
                    'avg_similarity': per_user_sim.get(uname),
                    'has_pin': has_pin,
                }

    # ── Risk band distribution ─────────────────────────────────────────────
    # Approximated from CSV: SPOOF=FAIL, matched=PASS, unmatched REAL=ambiguous
    risk_distribution = {
        'PASS': success_count,
        'CHALLENGE': 0,      # Not tracked in CSV yet (Phase 3 will refine)
        'FAIL': spoof_count,
    }

    metrics = {
        'total_attempts': total,
        'success_count': success_count,
        'fail_count': fail_count,
        'spoof_count': spoof_count,
        'success_rate': success_rate,
        'spoof_rate': spoof_rate,
        'risk_distribution': risk_distribution,
        'per_user_thresholds': per_user_thresholds,
        'recent_attempts': recent_list,
        'registered_users': state.identity_store.list_users() if state.identity_store else [],
        'cached_at': now,
    }

    # ── A/B comparison stats from live auth manager ─────────────────────
    if state.auth_mgr:
        ab_stats = state.auth_mgr.get_ab_stats()
        metrics['ab_comparison'] = {
            'false_reject_prevented': ab_stats.get('false_reject_prevented', 0),
            'security_tightened':     ab_stats.get('security_tightened', 0),
            'agreement_rate':         ab_stats.get('agreement_rate', 0),
            'total_ab_attempts':      ab_stats.get('total_ab_attempts', 0),
            'ab_enabled':             ab_stats.get('ab_enabled', False),
        }

    # Cache for 10 seconds
    _dashboard_cache['data'] = metrics
    _dashboard_cache['expires'] = now + 10.0

    return jsonify(sanitize_for_json(metrics))


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  Face Authentication Web Frontend")
    print("=" * 60)

    init_models()

    print("\n  Open in browser: http://localhost:5000")
    print("  Registration:    http://localhost:5000/register")
    print("  Home (unlock):   http://localhost:5000/")
    print("  Dashboard:       http://localhost:5000/dashboard")
    print("=" * 60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
