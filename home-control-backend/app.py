import logging
import crypto.nonce
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import Required, Length, EqualTo
from crypto.speck_wrapper import Speck
from websocket import create_connection
from utils.config import Config

# https://blog.miguelgrinberg.com/post/two-factor-authentication-with-flask
# https://github.com/miguelgrinberg/two-factor-auth-flask
# pip install flask-bootstrap
# pip install flask-wtf
# pip install flask-login
# pip install flask-sqlalchemy
# pip install websocket-client

actions = None
secure_actions = None
config_general = None
crypto_block_size = None
light_token_generator = None
config_home_control_server = None

# create application instance
app = Flask(__name__)
app.config.from_object('config')

# initialize extensions
bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
lm = LoginManager(app)

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True)
    password_hash = db.Column(db.String(128))

    def __init__(self, **kwargs):
        super(User, self).__init__(**kwargs)

    @property
    def password(self):
        raise AttributeError('password is not a readable attribute')

    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)

@lm.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[Required(), Length(1, 64)])
    password = PasswordField('Password', validators=[Required()])
    password_again = PasswordField('Password again', validators=[Required(), EqualTo('password')])
    submit = SubmitField('Register')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[Required(), Length(1, 64)])
    password = PasswordField('Password', validators=[Required()])
    submit = SubmitField('Login')

@app.route('/')
def index():
    return render_template('index.html', actions=actions)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        # if user is logged in we get out of here
        return redirect(url_for('index'))
    form = RegisterForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is not None:
            flash('Username already exists.')
            return redirect(url_for('register'))
        # add new user to the database
        user = User(username=form.username.data, password=form.password.data)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))    
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        # if user is logged in we get out of here
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.verify_password(form.password.data):
            flash('Invalid username or password.')
            return redirect(url_for('login'))
        login_user(user)
        flash('You are now logged in!')
        return redirect(url_for('index'))
    return render_template('login.html', form=form)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/action/<name>')
def action(name):
    return render_template('action.html', action=name) 

def check_authorization():
    logging.info("call check authorization")
    
    nonce = crypto.nonce.gen_nonce_sys()
    logging.info("nonce: " + nonce)
    
    client_ip = request.environ['REMOTE_ADDR']
    logging.info("client ip: " + client_ip)
    
    logging.info("create connection")
    socket = create_connection("ws://" + client_ip + ":5000/authorization") #  + ":5000/light_token"
    
    logging.info("send authorization request")
    socket.send("request_authorization:" + nonce)
    
    logging.info("wait for receive")
    response = socket.recv()
    socket.close()
    logging.info("response: " + response)
    
    key = light_token_generator.get_token()
    logging.info("token: " + key)
    
    logging.info("create speck cipher")
    speck = Speck(key, crypto_block_size)
    nonce = int(nonce)
    nonce += 1
    
    cipher = speck.encrypt(nonce)
    logging.info("cipher: " + cipher)
    
    authorization_result = (cipher == response)
    logging.info("authorization result: " + str(authorization_result))
    
    return authorization_result

@app.route("/api/evaluate/authorization")
def evaluate_authorization():
    return jsonify(result=check_authorization())

@app.route('/control/<int:action_id>')
def control(action_id):
    if action_id in secure_actions:
        valid_user = check_authorization()
        flash("special access rights required!")
        if valid_user:
            flash("special access granted")
            return redirect(url_for('action', name=actions[action_id]))
        else:
            flash("special access prohibited")
    else:
        flash("access granted!")
        return redirect(url_for('action', name=actions[action_id]))

@app.route('/shutdown')
def shutdown():
    logging.info("shutdown server")
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()    
    return jsonify(result=True)

def set_data(config_general_t, config_home_control_server_t, broadcast_light_token):
    global config_general
    global config_home_control_server
    global light_token_generator
    global crypto_block_size
    global actions
    global secure_actions
    
    #app.config['ini'] = config
    #app.config['light_token'] = broadcast_light_token
    #config_webserver = app.config['ini']
    config_general = config_general_t
    config_home_control_server = config_home_control_server_t
    light_token_generator = broadcast_light_token
    crypto_block_size = int(config_general.get("Crypto block size"))
    actions = Config.create_dict(config_home_control_server.get("Home actions"))
    secure_actions = Config.create_list(config_home_control_server.get("Home secure actions"))
    
def start():
    db.create_all()
    #config_webserver = app.config['ini']
    if config_general.get("tls") == "True":
        app.run(host=config_general.get("address"),
                port=int(config_general.get("port")),
                ssl_context=(config_general.get("certificate"),
                             config_general.get("key")))
    else:
        app.run(host=config_general.get("address"),
                port=int(config_general.get("port")))
