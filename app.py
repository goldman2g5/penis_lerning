

import pymysql
from flask import Flask, render_template, request, redirect, url_for
from pymysql.cursors import DictCursor

app = Flask(__name__)

dbh = pymysql.connect(
    host='185.12.94.106',
    user='2p1s23',
    password='941-961-748',
    db='2p1s23',
    charset='utf8mb4',
    cursorclass=DictCursor,
    autocommit=True
)


@app.route('/', methods=['post', 'get'])
def main():
    return render_template('glavnaya.html')


@app.route('/registration', methods=['post', 'get'])
def registration():

    message = 'Для регистрации заполните форму ниже'
    login, password, username = request.form.get('login'), request.form.get('password'), request.form.get(
        'username')  # запрос к данным формы

    if login and password and username and len(login) >= 4 and len(password) >= 4 and len(username) >= 4:
        cur = dbh.cursor()
        cur.execute(
            f'SELECT login, password, name FROM users WHERE login = "{login}" OR password = "{password}" OR name = "{username}";')
        b = cur.fetchone()

        if b is None:
            cur.execute(
                f'INSERT INTO users (id, login, password, name) VALUES (NULL, "{login}", "{password}", "{username}");')
            return redirect("/login")

        else:
            message = "Логин, пароль или имя уже используются, попробуйте другие"

    return render_template('reg.html', message=message)


@app.route('/login', methods=['post', 'get'])
def login():
    message = 'Для входа введите логин и пароль'
    login, password = request.form.get('username'), request.form.get('password')  # запрос к данным формы

    if login is not None or password is not None:
        cur = dbh.cursor()
        cur.execute(f'SELECT login, password FROM users WHERE login = "{login}" AND password = "{password}";')
        a = cur.fetchone()

        if a and a['login'] == login and a['password'] == password:
            return redirect("/neyronka")
        else:
            message = "Wrong username or password"
    return render_template('login.html')

@app.route('/neyronka', methods=['post', 'get'])
def neyronka():
    return render_template('neyronka.html')

if __name__ == "__main__":
    app.run(debug=True)
