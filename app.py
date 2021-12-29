#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pymysql
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request, redirect, flash
from keras.models import load_model
from pymysql.cursors import DictCursor
from werkzeug.utils import secure_filename

app = Flask(__name__)

dbh = pymysql.connect(
    host='185.12.94.106',
    user='2p2s10',
    password='231-429-617',
    db='2p2s10',
    charset='utf8mb4',
    cursorclass=DictCursor,
    autocommit=True
)

column = None
table = None
input_ = None
login = None
password = None
new_img = None
zalupka321 = None
filename = ""
tables_for_insert = []
values = []
pre_values = []
keys = []
keys_to_add = []
var = {1, 2}


@app.route('/', methods=['post', 'get'])
def main():
    return render_template('glavnaya.html')


@app.route('/registration', methods=['post', 'get'])
def registration():
    dbh = pymysql.connect(
        host='185.12.94.106',
        user='2p2s10',
        password='231-429-617',
        db='2p2s10',
        charset='utf8mb4',
        cursorclass=DictCursor,
        autocommit=True
    )
    message = 'Для регистрации заполните форму ниже'
    login, password = request.form.get('login'), request.form.get('password')  # запрос к данным формы

    if login and password:
        print(login, password)
        cur = dbh.cursor()
        cur.execute(
            f"SELECT login, password FROM users WHERE login = '{login}' OR password = '{password}';")
        huy = cur.fetchone()

        if huy is None:
            cur.execute(
                f"INSERT INTO users (login, password) VALUES ('{login}', '{password}');")
            return redirect("/login")

        else:
            message = "Логин, пароль или имя уже используются, попробуйте другие"

    return render_template('reg.html', message=message)


keys_list = []


@app.route('/login', methods=['post', 'get'])
def login():
    global dbh
    dbh = pymysql.connect(
        host='185.12.94.106',
        user='2p2s10',
        password='231-429-617',
        db='2p2s10',
        charset='utf8mb4',
        cursorclass=DictCursor,
        autocommit=True
    )
    message = 'Для входа введите логин и пароль'
    login, password = request.form.get('username'), request.form.get('password')  # запрос к данным формы

    if login is not None or password is not None:
        cur = dbh.cursor()
        cur.execute(f'SELECT login, password FROM users WHERE login = "{login}" AND password = "{password}";')
        a = cur.fetchone()
        print(a)

        if a and a['login'] == login and a['password'] == password:
            dbh = pymysql.connect(
                host='185.12.94.106',
                user=a['login'],
                password=a['password'],
                db=a['login'],
                charset='utf8mb4',
                cursorclass=DictCursor,
                autocommit=True
            )
            return redirect("/gigabaza")
        else:
            message = "Wrong username or password"
    return render_template('login.html', message=message)


@app.route('/garfield_race', methods=['post', 'get'])
def gonki():
    return render_template('gonki.html')


@app.route('/get_table', methods=['GET', 'POST'])
def get_name():
    global var
    global values
    global pre_values
    global keys
    global keys_to_add
    global tables_for_insert
    values = []
    pre_values = []
    keys = []
    keys_to_add = []
    table = request.form['zalupa']
    if table:
        cur = dbh.cursor()
        cur.execute(f"SELECT *FROM {table}")
        var = cur.fetchall()
        for i in var:
            keys = list(i.keys())
            keys_to_add = list(i.keys())
            pre_values = list(i.values())
            values.append(pre_values)
        tables_for_insert = keys_to_add
        keys_to_add = keys_to_add[1:]
    # for i in keys_to_add:
    #     if i == "admin_id" or :
    #         datatype.appemd('number')
    #     else:
    #         datatype.appemd('text')
    #         #словарь кейс иу адд и дататайп потом вывод одной переменной в последней функции вместо keys_to_add
    return render_template('baza.html')


@app.route('/get_input', methods=['GET', 'POST'])
def get_input():
    global new_name

    new_name = request.form['datas']
    new_name = new_name.split(",")

    keys_to_add = ', '.join(["'" + str(elem) + "'" for elem in new_name])
    print(keys_to_add)
    # сделать фильтр по кейс ту адд и в зависимоти от него присваивать тип данных или сделать валидацию в джес
    update_table = request.form['update_table']
    print(update_table)
    cur = dbh.cursor()
    cur.execute(f"SELECT *FROM {update_table}")
    ids = cur.fetchall()
    new_id = []
    for i in ids:
        z = i.get('id')
        z = int(z)
        new_id.append(z)
    new_id = new_id[-1]
    new_id = new_id + 1

    cur.execute(f'INSERT INTO {update_table} VALUES ("{new_id}", {keys_to_add});')

    return render_template('baza.html')


@app.route('/delete_user', methods=['GET', 'POST'])
def delete_user():
    global new_name
    users_to_delete = request.form['users_to_delete']
    users_to_delete = users_to_delete[1:]
    users_to_delete = users_to_delete.split(', ')
    users_to_delete = users_to_delete[0]
    update_table = request.form['update_table']
    cur = dbh.cursor()
    cur.execute(f'SET FOREIGN_KEY_CHECKS = 0;')
    cur.execute(f'DELETE FROM {update_table} where {update_table}. id = {users_to_delete}')
    print("Record deleted successfully")

    return render_template('baza.html')


@app.route('/gigabaza', methods=['post', 'get'])
def baza():
    cur = dbh.cursor()
    cur.execute("SHOW TABLES")
    tables = []
    for i in cur:
        tables.append(i.get(f"Tables_in_2p2s10"))
    print(tables)
    return render_template('baza.html', var=var, keys=keys, values=values, keys_to_add=keys_to_add, tables=tables)


UPLOAD_FOLDER = f'{os.path.abspath(os.getcwd())}'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
print(UPLOAD_FOLDER)


@app.route('/penis_learning', methods=['post', 'get'])
def neyronka():
    global new_img
    global zalupka321
    global filename

    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            new_img = UPLOAD_FOLDER + "/" + filename
            print(new_img)

        def exists(filePath):
            try:
                os.stat(filePath)
            except OSError:
                return False
            return True

        tf.random.set_seed(0)
        var = tf.keras.backend.clear_session

        emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

        def prepare_data(data):
            image_array = np.zeros(shape=(len(data), 48, 48))
            image_label = np.array(list(map(int, data['emotion'])))

            for i, row in enumerate(data.index):
                img = np.fromstring(data.loc[row, ' pixels'], dtype=int, sep=' ')
                img = np.reshape(img, (48, 48))
                image_array[i] = img

            return image_array, image_label

        def sample_plot(x, y=None):
            # x, y are numpy arrays
            n = 20
            samples = random.sample(range(x.shape[0]), n)

            figs, axs = plt.subplots(2, 10, figsize=(25, 5), sharex=True, sharey=True)
            ax = axs.ravel()
            for i in range(n):
                ax[i].imshow(x[samples[i], :, :], cmap=plt.get_cmap('gray'))
                ax[i].set_xticks([])
                ax[i].set_yticks([])
                if y is not None:
                    ax[i].set_title(emotions[y[samples[i]]])

        model = load_model("kaggle/model/models.h5")

        imgPath = new_img
        img = Image.open(imgPath).convert('L').resize((48, 48), Image.ANTIALIAS)
        img = np.array(img)

        prediction = model.predict(img[None, :, :])
        zalupka321 = "".join(list(emotions[np.argmax(prediction)]))
        print("Распознан объект: ", zalupka321)
    return render_template('neyronka.html', zalupka321=zalupka321, new_img=UPLOAD_FOLDER + "\\" + filename)


if __name__ == "__main__":
    app.run(debug=True)

# cur.execute(f"SELECT admins.id, admins.name, room.id, room.room_name FROM admins, room, room_rent, console, guests")
# room.console_id, room.admin_id, room_rent.id, room_rent.rent_time, room_rent.guest_id, room_rent.room_id
